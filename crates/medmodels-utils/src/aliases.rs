pub type MrHashMap<K, V> = hashbrown::HashMap<K, V>;
pub type MrHashMapEntry<'a, K, V, S> = hashbrown::hash_map::Entry<'a, K, V, S>;
pub type MrHashSet<T> = hashbrown::HashSet<T>;
