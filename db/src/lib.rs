#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::missing_errors_doc)]
#![forbid(unsafe_code)]
use chrono::{DateTime, Utc};
use rocksdb::{
    ColumnFamily, ColumnFamilyDescriptor, IteratorMode, Options, Transaction, TransactionDB,
    TransactionDBOptions,
};
use std::collections::{BTreeMap, BTreeSet};
use std::marker::PhantomData;
use std::path::Path;
use std::time::Duration;

type BincodeConfigType =
    bincode::config::Configuration<bincode::config::BigEndian, bincode::config::Fixint>;

const BINCODE_CONFIG: BincodeConfigType = bincode::config::standard()
    .with_big_endian()
    .with_fixed_int_encoding();

const QUEUE_CF_NAME: &str = "queue";
const LOOKUP_CF_NAME: &str = "lookup";
const LOG_CF_NAME: &str = "log";

#[derive(thiserror::Error, Debug)]
pub enum Error<I, P> {
    #[error("RocksDB error")]
    RocksDb(#[from] rocksdb::Error),
    #[error("Invalid ID")]
    InvalidId(I),
    #[error("Invalid priority")]
    InvalidPriority(P),
    #[error("Invalid timestamp")]
    InvalidTimestamp(DateTime<Utc>),
    #[error("Invalid timestamp second")]
    InvalidTimestampSecond(u32),
    #[error("Invalid queue key")]
    InvalidQueueKey(Vec<u8>),
    #[error("Invalid queue value")]
    InvalidQueueValue(Vec<u8>),
    #[error("Invalid lookup key")]
    InvalidLookupKey(Vec<u8>),
    #[error("Invalid lookup value")]
    InvalidLookupValue(Vec<u8>),
    #[error("Invalid log key")]
    InvalidLogKey(Vec<u8>),
    #[error("Missing queue entry")]
    MissingQueueEntry(P, I),
}

pub trait Priority: Sized {
    fn encode_priority(&self) -> Option<u64>;
    fn decode_priority(value: u64) -> Option<Self>;
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct QueueEntry<P> {
    pub priority: P,
    pub expiration: Option<DateTime<Utc>>,
}

impl<P> QueueEntry<P> {
    pub fn new(priority: P, expiration: Option<DateTime<Utc>>) -> Self {
        Self {
            priority,
            expiration,
        }
    }
}

pub struct PrioritiesDb<I, P> {
    db: TransactionDB,
    _format: PhantomData<(I, P)>,
}

impl<I: bincode::Encode + bincode::Decode<()>, P: Priority> PrioritiesDb<I, P> {
    /// Insert a prioritized ID into the database.
    ///
    /// Optionally returns the previous entry for the ID, if there was one.
    pub fn insert(
        &self,
        id: &I,
        priority: &P,
        retrieve_previous: bool,
    ) -> Result<Option<QueueEntry<P>>, Error<I, P>>
    where
        I: Clone,
        P: Clone,
    {
        let priority_bytes = priority
            .encode_priority()
            .ok_or_else(|| Error::InvalidPriority(priority.clone()))?
            .to_be_bytes();
        let id_bytes = bincode::encode_to_vec(&id, BINCODE_CONFIG)
            .map_err(|_| Error::InvalidId(id.clone()))?;

        let queue_key_bytes = Self::queue_key_bytes(&priority_bytes, &id_bytes);

        let queue_cf = self.queue_cf();
        let lookup_cf = self.lookup_cf();

        let transaction = self.db.transaction();
        let previous = if retrieve_previous {
            let previous =
                Self::lookup_queue_entry(&transaction, queue_cf, lookup_cf, &id, &id_bytes)?;

            if let Some((queue_key_bytes, _)) = &previous {
                transaction.delete_cf(queue_cf, queue_key_bytes)?;
            }

            previous.map(|(_, entry)| entry)
        } else {
            None
        };

        transaction.put_cf(queue_cf, queue_key_bytes, [])?;
        transaction.put_cf(lookup_cf, id_bytes, priority_bytes)?;
        transaction.commit()?;

        Ok(previous)
    }

    pub fn reserve_next(
        &self,
        count: usize,
        reservation: Duration,
        now: DateTime<Utc>,
    ) -> Result<Vec<I>, Error<I, P>> {
        let expiration = now + reservation;
        let expiration_bytes = Self::encode_timestamp(expiration)?;
        let queue_cf = self.queue_cf();

        let transaction = self.db.transaction();
        let mut ids = vec![];

        for result in transaction.iterator_cf(queue_cf, IteratorMode::Start) {
            let (key_bytes, value_bytes) = result?;

            let timestamp = Self::decode_queue_value(value_bytes)?;

            if timestamp.is_none_or(|timestamp| timestamp <= now) {
                if key_bytes.len() < 8 {
                    return Err(Error::InvalidQueueKey(key_bytes.to_vec()));
                }

                let id_bytes = &key_bytes[8..];

                if id_bytes.len() != 8 {
                    return Err(Error::InvalidQueueKey(key_bytes.to_vec()));
                }

                let (id, read) = bincode::decode_from_slice::<I, _>(id_bytes, BINCODE_CONFIG)
                    .map_err(|_| Error::InvalidQueueKey(key_bytes.to_vec()))?;

                if read != 8 {
                    return Err(Error::InvalidQueueKey(key_bytes.to_vec()));
                }

                transaction.put_cf(queue_cf, key_bytes, expiration_bytes)?;

                ids.push(id);

                if ids.len() == count {
                    break;
                }
            }
        }

        transaction.commit()?;

        Ok(ids)
    }

    pub fn cancel_reservations<'a, IS: Iterator<Item = &'a I>>(
        &self,
        ids: IS,
    ) -> Result<Vec<Option<QueueEntry<P>>>, Error<I, P>>
    where
        I: Clone + 'a,
        P: Clone,
    {
        let queue_cf = self.queue_cf();
        let lookup_cf = self.lookup_cf();

        let transaction = self.db.transaction();

        let queue_entries = ids
            .map(|id| {
                let id_bytes = bincode::encode_to_vec(id, BINCODE_CONFIG)
                    .map_err(|_| Error::InvalidId(id.clone()))?;

                let queue_entry =
                    Self::lookup_queue_entry(&transaction, queue_cf, lookup_cf, id, &id_bytes)?;

                // If there is an entry, we clear its reservation.
                if let Some((queue_key_bytes, _)) = &queue_entry {
                    transaction.put_cf(queue_cf, queue_key_bytes, [])?;
                }

                Ok(queue_entry.map(|(_, entry)| entry))
            })
            .collect::<Result<Vec<_>, Error<I, P>>>()?;

        transaction.commit()?;

        Ok(queue_entries)
    }

    pub fn complete<'a, IS: Iterator<Item = &'a (I, Option<P>)>>(
        &self,
        ids: IS,
        timestamp: DateTime<Utc>,
    ) -> Result<Vec<Option<QueueEntry<P>>>, Error<I, P>>
    where
        I: Clone + 'a,
        P: Clone + 'a,
    {
        let queue_cf = self.queue_cf();
        let lookup_cf = self.lookup_cf();
        let log_cf = self.log_cf();

        let transaction = self.db.transaction();

        let queue_entries = ids
            .map(|(id, priority)| {
                let id_bytes = bincode::encode_to_vec(id, BINCODE_CONFIG)
                    .map_err(|_| Error::InvalidId(id.clone()))?;

                let queue_entry =
                    Self::lookup_queue_entry(&transaction, queue_cf, lookup_cf, id, &id_bytes)?;

                if let Some((queue_key_bytes, _)) = &queue_entry {
                    transaction.delete_cf(queue_cf, queue_key_bytes)?;
                }

                if let Some(priority) = priority {
                    let priority_bytes = priority
                        .encode_priority()
                        .ok_or_else(|| Error::InvalidPriority(priority.clone()))?
                        .to_be_bytes();
                    let queue_key_bytes = Self::queue_key_bytes(&priority_bytes, &id_bytes);

                    let mut log_key_bytes = Vec::with_capacity(id_bytes.len() + 4);
                    log_key_bytes.extend(&id_bytes);
                    log_key_bytes.extend(&Self::encode_timestamp_reverse(timestamp)?);

                    transaction.put_cf(queue_cf, queue_key_bytes, [])?;
                    transaction.put_cf(lookup_cf, id_bytes, priority_bytes)?;
                    transaction.put_cf(log_cf, log_key_bytes, [])?;
                }

                Ok(queue_entry.map(|(_, entry)| entry))
            })
            .collect::<Result<Vec<_>, Error<I, P>>>()?;

        transaction.commit()?;

        Ok(queue_entries)
    }

    pub fn id_log(&self, id: &I) -> Result<Vec<DateTime<Utc>>, Error<I, P>>
    where
        I: Clone,
    {
        let log_cf = self.log_cf();

        let id_bytes = bincode::encode_to_vec(&id, BINCODE_CONFIG)
            .map_err(|_| Error::InvalidId(id.clone()))?;

        let mut timestamps = vec![];

        for result in self.db.iterator_cf(
            log_cf,
            IteratorMode::From(&id_bytes, rocksdb::Direction::Forward),
        ) {
            let (log_key_bytes, _) = result?;

            if log_key_bytes.starts_with(&id_bytes) {
                if log_key_bytes.len() == id_bytes.len() + 4 {
                    let timestamp_s = u32::from_be_bytes(
                        log_key_bytes[id_bytes.len()..id_bytes.len() + 4]
                            .try_into()
                            .map_err(|_| Error::InvalidLogKey(log_key_bytes.to_vec()))?,
                    );
                    let timestamp = Self::decode_timestamp_reverse(timestamp_s)?;

                    timestamps.push(timestamp);
                } else {
                    return Err(Error::InvalidLogKey(log_key_bytes.to_vec()));
                }
            } else {
                break;
            }
        }

        Ok(timestamps)
    }

    fn lookup_priority(
        transaction: &Transaction<TransactionDB>,
        lookup_cf: &ColumnFamily,
        id_bytes: &[u8],
    ) -> Result<Option<P>, Error<I, P>> {
        match transaction.get_pinned_for_update_cf(lookup_cf, id_bytes, true)? {
            Some(lookup_value_bytes) => {
                let priority_value = u64::from_be_bytes(
                    lookup_value_bytes
                        .as_ref()
                        .try_into()
                        .map_err(|_| Error::InvalidLookupValue(lookup_value_bytes.to_vec()))?,
                );

                let priority = P::decode_priority(priority_value)
                    .ok_or_else(|| Error::InvalidLookupValue(lookup_value_bytes.to_vec()))?;

                Ok(Some(priority))
            }
            None => Ok(None),
        }
    }

    /// Look up an entry by ID.
    ///
    /// Returns the queue key and entry if one exists.
    fn lookup_queue_entry(
        transaction: &Transaction<TransactionDB>,
        queue_cf: &ColumnFamily,
        lookup_cf: &ColumnFamily,
        id: &I,
        id_bytes: &[u8],
    ) -> Result<Option<(Vec<u8>, QueueEntry<P>)>, Error<I, P>>
    where
        P: Clone,
        I: Clone,
    {
        Self::lookup_priority(transaction, lookup_cf, id_bytes)?
            .map(|priority| {
                let priority_bytes = priority
                    .encode_priority()
                    .ok_or_else(|| Error::InvalidPriority(priority.clone()))?
                    .to_be_bytes();
                let queue_key_bytes = Self::queue_key_bytes(&priority_bytes, id_bytes);

                let queue_value_bytes =
                    transaction.get_pinned_for_update_cf(queue_cf, &queue_key_bytes, true)?;

                match queue_value_bytes {
                    Some(queue_value_bytes) => {
                        let expiration = if queue_value_bytes.is_empty() {
                            None
                        } else {
                            let expiration_s =
                                u32::from_be_bytes(queue_value_bytes.as_ref().try_into().map_err(
                                    |_| Error::InvalidQueueValue(queue_value_bytes.to_vec()),
                                )?);

                            Some(Self::decode_timestamp(expiration_s)?)
                        };

                        Ok((
                            queue_key_bytes,
                            QueueEntry {
                                priority,
                                expiration,
                            },
                        ))
                    }
                    None => Err(Error::MissingQueueEntry(priority, id.clone())),
                }
            })
            .map_or_else(|| Ok(None), |result| result.map(Some))
    }

    fn queue_key_bytes(priority_bytes: &[u8], id_bytes: &[u8]) -> Vec<u8> {
        let mut queue_key_bytes = Vec::with_capacity(8 + id_bytes.len());
        queue_key_bytes.extend(priority_bytes);
        queue_key_bytes.extend(id_bytes);
        queue_key_bytes
    }

    fn encode_timestamp(timestamp: DateTime<Utc>) -> Result<[u8; 4], Error<I, P>> {
        u32::try_from(timestamp.timestamp())
            .map(u32::to_be_bytes)
            .map_err(|_| Error::InvalidTimestamp(timestamp))
    }

    fn decode_timestamp(timestamp_s: u32) -> Result<DateTime<Utc>, Error<I, P>> {
        DateTime::from_timestamp(timestamp_s.into(), 0)
            .ok_or_else(|| Error::InvalidTimestampSecond(timestamp_s))
    }

    fn encode_timestamp_reverse(timestamp: DateTime<Utc>) -> Result<[u8; 4], Error<I, P>> {
        u32::try_from(timestamp.timestamp())
            .map(|timestamp_s| (u32::MAX - timestamp_s).to_be_bytes())
            .map_err(|_| Error::InvalidTimestamp(timestamp))
    }

    fn decode_timestamp_reverse(timestamp_s: u32) -> Result<DateTime<Utc>, Error<I, P>> {
        Self::decode_timestamp(u32::MAX - timestamp_s)
    }

    fn decode_queue_value<B: AsRef<[u8]>>(bytes: B) -> Result<Option<DateTime<Utc>>, Error<I, P>> {
        if bytes.as_ref().is_empty() {
            Ok(None)
        } else {
            let timestamp_s = u32::from_be_bytes(
                bytes
                    .as_ref()
                    .try_into()
                    .map_err(|_| Error::InvalidQueueValue(bytes.as_ref().to_vec()))?,
            );

            Self::decode_timestamp(timestamp_s).map(Some)
        }
    }
}

impl<I, P> PrioritiesDb<I, P> {
    pub fn open<D: AsRef<Path>>(directory: D) -> Result<Self, rocksdb::Error> {
        let queue_cf = ColumnFamilyDescriptor::new(QUEUE_CF_NAME, Options::default());
        let lookup_cf = ColumnFamilyDescriptor::new(LOOKUP_CF_NAME, Options::default());
        let log_cf = ColumnFamilyDescriptor::new(LOOKUP_CF_NAME, Options::default());

        let cfs = vec![queue_cf, lookup_cf, log_cf];

        let mut options = Options::default();
        options.create_missing_column_families(true);
        options.create_if_missing(true);

        let transaction_options = TransactionDBOptions::default();

        let db =
            TransactionDB::open_cf_descriptors(&options, &transaction_options, directory, cfs)?;

        Ok(Self {
            db,
            _format: PhantomData,
        })
    }

    fn queue_cf(&self) -> &ColumnFamily {
        self.db
            .cf_handle(QUEUE_CF_NAME)
            .expect("Queue table column family does not exist")
    }

    fn lookup_cf(&self) -> &ColumnFamily {
        self.db
            .cf_handle(LOOKUP_CF_NAME)
            .expect("Lookup table column family does not exist")
    }

    fn log_cf(&self) -> &ColumnFamily {
        self.db
            .cf_handle(LOG_CF_NAME)
            .expect("Log table column family does not exist")
    }
}

#[cfg(test)]
mod tests {
    use super::{PrioritiesDb, Priority, QueueEntry};
    use chrono::{DateTime, SubsecRound, Utc};
    use std::time::Duration;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct CodeTimestamp {
        code: u32,
        timestamp: DateTime<Utc>,
    }

    impl CodeTimestamp {
        fn new(code: u32, timestamp: DateTime<Utc>) -> Self {
            Self {
                code,
                timestamp: timestamp.round_subsecs(0),
            }
        }
    }

    impl Priority for CodeTimestamp {
        fn encode_priority(&self) -> Option<u64> {
            let timestamp_s: u32 = self.timestamp.timestamp().try_into().ok()?;

            let mut bytes = [0u8; 8];
            bytes[0..4].copy_from_slice(&self.code.to_be_bytes());
            bytes[4..8].copy_from_slice(&timestamp_s.to_be_bytes());

            Some(u64::from_be_bytes(bytes))
        }

        fn decode_priority(value: u64) -> Option<Self> {
            let bytes = value.to_be_bytes();
            let code_bytes: [u8; 4] = bytes[0..4].try_into().ok()?;
            let timestamp_bytes: [u8; 4] = bytes[4..8].try_into().ok()?;
            let timestamp =
                DateTime::from_timestamp(u32::from_be_bytes(timestamp_bytes).into(), 0)?;

            Some(Self {
                code: u32::from_be_bytes(code_bytes),
                timestamp,
            })
        }
    }

    #[test]
    fn reserve_next() -> Result<(), Box<dyn std::error::Error>> {
        let test_db_dir = tempfile::tempdir().unwrap();

        let db: PrioritiesDb<u64, CodeTimestamp> = PrioritiesDb::open(test_db_dir)?;
        let now = Utc::now();
        let timestamp_a = now + chrono::Duration::seconds(10);
        let timestamp_b = now + chrono::Duration::seconds(20);

        let priority_0 = CodeTimestamp::new(0, timestamp_a);
        let priority_1 = CodeTimestamp::new(0, timestamp_b);
        let priority_2 = CodeTimestamp::new(1, timestamp_a);

        let previous = db.insert(&987, &priority_0, true)?;
        assert_eq!(previous, None);

        let previous = db.insert(&123, &priority_0, true)?;
        assert_eq!(previous, None);

        let previous = db.insert(&123, &priority_1, true)?;
        assert_eq!(previous, Some(QueueEntry::new(priority_0, None)));

        let previous = db.insert(&456, &priority_2, true)?;
        assert_eq!(previous, None);

        let next = db.reserve_next(10, Duration::from_secs(30), now)?;

        assert_eq!(next, vec![987, 123, 456]);

        Ok(())
    }
}
