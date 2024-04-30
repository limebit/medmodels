use std::{collections::HashMap, hash::Hash};

pub trait DeepFrom<T> {
    fn deep_from(value: T) -> Self;
}

pub trait DeepInto<T> {
    fn deep_into(self) -> T;
}

impl<T, F> DeepInto<T> for F
where
    T: DeepFrom<F>,
{
    fn deep_into(self) -> T {
        T::deep_from(self)
    }
}

impl<K, KF, V, VF> DeepFrom<HashMap<K, V>> for HashMap<KF, VF>
where
    KF: Hash + Eq + DeepFrom<K>,
    VF: DeepFrom<V>,
{
    fn deep_from(value: HashMap<K, V>) -> Self {
        value
            .into_iter()
            .map(|(key, value)| (key.deep_into(), value.deep_into()))
            .collect()
    }
}

impl<K, KF, V, VF> DeepFrom<(K, V)> for (KF, VF)
where
    KF: DeepFrom<K>,
    VF: DeepFrom<V>,
{
    fn deep_from(value: (K, V)) -> Self {
        (value.0.deep_into(), value.1.deep_into())
    }
}

impl<K, KF, V, VF> DeepFrom<(K, K, V)> for (KF, KF, VF)
where
    KF: DeepFrom<K>,
    VF: DeepFrom<V>,
{
    fn deep_from(value: (K, K, V)) -> Self {
        (
            value.0.deep_into(),
            value.1.deep_into(),
            value.2.deep_into(),
        )
    }
}

impl<V, VF> DeepFrom<Vec<V>> for Vec<VF>
where
    VF: DeepFrom<V>,
{
    fn deep_from(value: Vec<V>) -> Self {
        value.into_iter().map(VF::deep_from).collect()
    }
}

impl<V, VF> DeepFrom<Option<V>> for Option<VF>
where
    VF: DeepFrom<V>,
{
    fn deep_from(value: Option<V>) -> Self {
        value.map(|v| v.deep_into())
    }
}
