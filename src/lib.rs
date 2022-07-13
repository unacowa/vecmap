#![feature(test)]
extern crate test;

use std::sync::{Arc, Mutex};
use std::sync::{RwLock, RwLockWriteGuard, RwLockReadGuard};
use std::ops::{Deref, DerefMut};

/// Vector Map Builder
///
/// To use VecMap, specify the size of the required Vector
/// and allocate memory after specifying all of them.
///
/// VecMapBuilder provides the above steps.
pub struct VecMapBuilder {
    partitions: Arc<Mutex<Vec<isize>>>,
}

impl VecMapBuilder {

    /// Create new storage builder.
    pub fn new() -> Self {
	VecMapBuilder { partitions: Arc::new(Mutex::new(vec![0])) }
    }

    /// Allocate memory buffer with size.
    /// This phase does not really allocate memory.
    /// Allocation is thread-safe, but does not improve performance
    /// because of the global mutex.
    ///
    pub fn alloc(&self, size: isize) -> usize {
	let mut partitions = self.partitions.lock().unwrap();
	let len = partitions.len();
	let offset = partitions[len - 1];
	partitions.push(offset + size);
	len - 1
    }

    /// Build mutable vecmap, and really allocate all memory.
    /// # Examples
    ///
    /// ```
    /// use vecmap::VecMapBuilder;
    /// 
    /// let builder = VecMapBuilder::new();
    /// builder.alloc(10);
    /// builder.alloc(5);
    /// let vecmap = builder.build_mut(0); // 0 is a initial value.
    /// ```
    pub fn build_mut<T: Clone>(&self, initial_value: T) -> MutVecMap<T> {
	let partitions = self.partitions.lock().unwrap().iter()
	    .map(|x| RwLock::new(*x))
	    .collect::<Vec<_>>();
	let len = *partitions.last().unwrap().read().unwrap();
	let store = vec![initial_value; len as usize];
	MutVecMap {store, partitions}
    }

    /// Build immutable vecmap, and really allocate all memory.
    /// # Examples
    ///
    /// ```
    /// use vecmap::VecMapBuilder;
    /// 
    /// let builder = VecMapBuilder::new();
    /// builder.alloc(10);
    /// builder.alloc(5);
    /// let vecmap = builder.build(0); // 0 is a initial value.
    /// ```
    pub fn build<T: Clone>(&self, initial_value: T) -> VecMap<T> {
	let partitions = self.partitions.lock().unwrap().iter()
	    .map(|x| *x)
	    .collect::<Vec<_>>();
	let len = *partitions.last().unwrap();
	let store = vec![initial_value; len as usize];
	VecMap {store, partitions}
    }
}

/// (Immutable) Vector Mapping
///
/// Immutable version of Mutable Vector Map.
/// 
pub struct VecMap<T> {
    store: Vec<T>,
    partitions: Vec<isize>,
}

impl<T: Clone + Sized> VecMap<T> {
    pub fn borrow(&self, idx: usize) -> &[T] {
	let offset = self.partitions[idx];
	let size = self.partitions[idx + 1] - offset;
	let slice = unsafe {
	    let ptr = self.store.as_ptr().offset(offset) as *mut T;
	    std::slice::from_raw_parts(ptr, size as usize)
	};
	slice
    }
}


/// Mutable Vector Mapping
///
/// MutVecMap has the ability to return a slice of the specified index
/// from the serially allocated memory.
/// 
/// It does not have separate length/capacity as std::Vec does,
/// since the memory is serially allocated.
/// Therefore, the amount of memory required is
/// almost the same as the number of elements.
///
/// A common idiom in Rust is to use `Arc<RwLock<Vec<_>>>`
/// when you need a thread-safe vector,
/// which requires 72 bytes of memory (Arc: 8, Mutex: 40, Vec: 24)
/// in addition to the vector contents.
///
/// For VecMap, RwLock : 40, isize: 8 for a total of 48Byte.
pub struct MutVecMap<T> {
    store: Vec<T>,
    partitions: Vec<RwLock<isize>>,
}

impl<T: Clone + Sized> MutVecMap<T> {
    /// borrow returns immutable slice of chunk.
    /// Wrapperd by SliceReadGuard.
    ///
    /// # Example
    ///
    /// ```
    /// use vecmap::VecMapBuilder;
    /// 
    /// let builder = VecMapBuilder::new();
    /// builder.alloc(10);
    /// builder.alloc(5);
    /// let vecmap = builder.build_mut(0);
    /// let s0 = vecmap.borrow(0);
    /// assert_eq!(s0.len(), 10);
    /// ```
    pub fn borrow(&self, idx: usize) -> SliceReadGuard<T> {
	let guard = self.partitions[idx].read().unwrap();
	let offset = *guard;
	let size = *self.partitions[idx + 1].read().unwrap() - offset;
	let slice = unsafe {
	    let ptr = self.store.as_ptr().offset(offset) as *mut T;
	    std::slice::from_raw_parts(ptr, size as usize)
	};
	SliceReadGuard { guard, slice }
    }
    
    /// borrow_mut returns mutable slice of chunk.
    /// Wrapperd by SliceWriteGuard.
    /// 
    /// # Example
    ///
    /// ```
    /// use vecmap::VecMapBuilder;
    /// 
    /// let builder = VecMapBuilder::new();
    /// builder.alloc(10);
    /// builder.alloc(5);
    /// let vecmap = builder.build_mut(0);
    /// let mut s1 = vecmap.borrow_mut(1);
    /// s1[1] = 9;
    /// ```
    pub fn borrow_mut(&self, idx: usize) -> SliceWriteGuard<T> {
	let guard = self.partitions[idx].write().unwrap();
	let offset = *guard;
	let size = *self.partitions[idx + 1].read().unwrap() - offset;
	let slice = unsafe {
	    let ptr = self.store.as_ptr().offset(offset) as *mut T;
	    std::slice::from_raw_parts_mut(ptr, size as usize)
	};
	SliceWriteGuard { guard, slice }
    }
}

#[derive(Debug)]
pub struct SliceReadGuard<'a, T> {
    guard: RwLockReadGuard<'a, isize>,
    slice: &'a [T],
}

impl<T> Deref for SliceReadGuard<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &*self.slice
    }
}

impl<T> Drop for SliceReadGuard<'_, T> {
    #[inline]
    fn drop(&mut self) {
	drop(&mut self.guard);
    }
}

#[derive(Debug)]
pub struct SliceWriteGuard<'a, T> {
    guard: RwLockWriteGuard<'a, isize>,
    slice: &'a mut [T],
}

impl<T> Deref for SliceWriteGuard<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &*self.slice
    }
}

impl<T> DerefMut for SliceWriteGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut *self.slice
    }
}

impl<T> Drop for SliceWriteGuard<'_, T> {
    #[inline]
    fn drop(&mut self) {
	drop(&mut self.guard);
    }
}


unsafe impl<T> Send for MutVecMap<T> {}
unsafe impl<T> Sync for MutVecMap<T> {}
unsafe impl<T> Send for VecMap<T> {}
unsafe impl<T> Sync for VecMap<T> {}


#[cfg(test)]
mod vac_store_tests {
    use super::*;
    use test::Bencher;
    use rayon::prelude::*;
    use rand::seq::SliceRandom;
    
    #[test]
    fn read_concurrent() {
	let n = 1000;
	let m = 100;
	let vsbuilder = VecMapBuilder::new();
	let allocs = (1..n).collect::<Vec<isize>>();
	let allocated: Arc<Vec<usize>> = Arc::new(
	    allocs.into_par_iter().map(|x| vsbuilder.alloc(x)).collect());

	let vs = vsbuilder.build(1);
	
	(0..m).collect::<Vec<isize>>().par_iter()
	    .for_each(|_| {
		for &idx in allocated.iter() {
		    let slice = vs.borrow(idx);
		    assert!(slice.iter().all(|&x| x == 1));
		}
	    });
    }

    #[bench]
    fn bench_concurrent_read(b: &mut Bencher) {
	let n = 10000;
	let m = 10;
	let vsbuilder = VecMapBuilder::new();
	let allocs = (1..n).collect::<Vec<isize>>();
	let allocated: Arc<Vec<usize>> = Arc::new(
	    allocs.into_par_iter().map(|x| vsbuilder.alloc(x)).collect());

	let vs = vsbuilder.build(1);

	let mut rng = rand::thread_rng();
	let xss = (0..m).map(|_| {
	    let mut xs = allocated.to_vec();
	    xs.shuffle(&mut rng);
	    xs
	}).collect::<Vec<_>>();
	
        b.iter(|| {
	    xss.par_iter().for_each(|xs| {
		for &idx in xs.iter() {
		    let slice = vs.borrow(idx);
		    slice.iter().fold(0, |x, y| x + y);
		}
	    });
	});
    }

    #[bench]
    fn bench_serialized_read(b: &mut Bencher) {
	let n = 10000;
	let m = 10;
	let vsbuilder = VecMapBuilder::new();
	let allocs = (1..n).collect::<Vec<isize>>();
	let allocated: Arc<Vec<usize>> = Arc::new(
	    allocs.into_par_iter().map(|x| vsbuilder.alloc(x)).collect());

	let vs = vsbuilder.build(1);

	let mut rng = rand::thread_rng();
	let xss = (0..m).map(|_| {
	    let mut xs = allocated.to_vec();
	    xs.shuffle(&mut rng);
	    xs
	}).collect::<Vec<_>>();
	
        b.iter(|| {
	    xss.iter().for_each(|xs| {
		for &idx in xs.iter() {
		    let slice = vs.borrow(idx);
		    slice.iter().fold(0, |x, y| x + y);
		}
	    });
	});
    }
}


#[cfg(test)]
mod mut_vac_store_tests {
    use super::*;
    use test::Bencher;
    use rayon::prelude::*;
    use rand::seq::SliceRandom;
    
    #[test]
    fn builder_allocates_concurrent() {
	let n = 10000;
	let vsbuilder = VecMapBuilder::new();
	let allocs = (1..n).collect::<Vec<isize>>();
	let sum_allocs = allocs.iter().sum::<isize>();
	let num_allocs = allocs.len();
	let mut allocated: Vec<usize> = allocs.into_par_iter()
	    .map(|x| vsbuilder.alloc(x)).collect();
	allocated.sort();

	let vs = vsbuilder.build_mut(0.0);
	assert_eq!(sum_allocs, vs.store.len() as isize);
	assert_eq!(num_allocs, allocated.len());
    }

    #[test]
    fn update_concurrent() {
	let n = 1000;
	let m = 100;
	let vsbuilder = VecMapBuilder::new();
	let allocs = (1..n).collect::<Vec<isize>>();
	let allocated: Arc<Vec<usize>> = Arc::new(
	    allocs.into_par_iter().map(|x| vsbuilder.alloc(x)).collect());

	let vs = vsbuilder.build_mut(0);
	
	(0..m).collect::<Vec<isize>>().par_iter()
	    .for_each(|_| {
		for &idx in allocated.iter() {
		    let mut slice = vs.borrow_mut(idx);
		    slice.iter_mut().for_each(|x| *x += 1);
		}
	    });
	assert!(vs.store.iter().all(|&x| x == m));
    }

    #[test]
    fn read_concurrent() {
	let n = 1000;
	let m = 100;
	let vsbuilder = VecMapBuilder::new();
	let allocs = (1..n).collect::<Vec<isize>>();
	let allocated: Arc<Vec<usize>> = Arc::new(
	    allocs.into_par_iter().map(|x| vsbuilder.alloc(x)).collect());

	let vs = vsbuilder.build_mut(1);
	
	(0..m).collect::<Vec<isize>>().par_iter()
	    .for_each(|_| {
		for &idx in allocated.iter() {
		    let slice = vs.borrow(idx);
		    assert!(slice.iter().all(|&x| x == 1));
		}
	    });
    }
    
    #[bench]
    fn bench_concurrent_alloc(b: &mut Bencher) {
	let n = 10000;
	let vsbuilder = VecMapBuilder::new();
        b.iter(|| {
	    let allocs = (1..n).collect::<Vec<isize>>();
	    allocs.into_par_iter()
		.for_each(|x| { vsbuilder.alloc(x); });
	});
    }    

    #[bench]
    fn bench_serialized_alloc(b: &mut Bencher) {
	let n = 10000;
	let vsbuilder = VecMapBuilder::new();
        b.iter(|| {
	    let allocs = (1..n).collect::<Vec<isize>>();
	    allocs.into_iter()
		.for_each(|x| { vsbuilder.alloc(x); });
	});
    }
    
    #[bench]
    fn bench_concurrent_update(b: &mut Bencher) {
	let n = 1000;
	let m = 100;
	let vsbuilder = VecMapBuilder::new();
	let allocs = (1..n).collect::<Vec<isize>>();
	let allocated: Arc<Vec<usize>> = Arc::new(
	    allocs.into_par_iter().map(|x| vsbuilder.alloc(x)).collect());

	let vs = vsbuilder.build_mut(0);

	let mut rng = rand::thread_rng();
	let xss = (0..m).map(|_| {
	    let mut xs = allocated.to_vec();
	    xs.shuffle(&mut rng);
	    xs
	}).collect::<Vec<_>>();
	
        b.iter(|| {
	    xss.par_iter().for_each(|xs| {
		for &idx in xs.iter() {
		    let mut slice = vs.borrow_mut(idx);
		    slice.iter_mut().for_each(|x| *x += 1);
		}
	    });
	});
    }

    #[bench]
    fn bench_concurrent_read(b: &mut Bencher) {
	let n = 1000;
	let m = 100;
	let vsbuilder = VecMapBuilder::new();
	let allocs = (1..n).collect::<Vec<isize>>();
	let allocated: Arc<Vec<usize>> = Arc::new(
	    allocs.into_par_iter().map(|x| vsbuilder.alloc(x)).collect());

	let vs = vsbuilder.build_mut(1);

	let mut rng = rand::thread_rng();
	let xss = (0..m).map(|_| {
	    let mut xs = allocated.to_vec();
	    xs.shuffle(&mut rng);
	    xs
	}).collect::<Vec<_>>();
	
        b.iter(|| {
	    xss.par_iter().for_each(|xs| {
		for &idx in xs.iter() {
		    let slice = vs.borrow(idx);
		    slice.iter().fold(0, |x, y| x + y);
		}
	    });
	});
    }
    
    #[bench]
    fn bench_serialized_update(b: &mut Bencher) {
	let n = 1000;
	let m = 100;
	let vsbuilder = VecMapBuilder::new();
	let allocs = (1..n).collect::<Vec<isize>>();
	let allocated: Arc<Vec<usize>> = Arc::new(
	    allocs.into_par_iter().map(|x| vsbuilder.alloc(x)).collect());

	let vs = vsbuilder.build_mut(0);

	let mut rng = rand::thread_rng();
	let xss = (0..m).map(|_| {
	    let mut xs = allocated.to_vec();
	    xs.shuffle(&mut rng);
	    xs
	}).collect::<Vec<_>>();
	
        b.iter(|| {
	    xss.iter().for_each(|xs| {
		for &idx in xs.iter() {
		    let mut slice = vs.borrow_mut(idx);
		    slice.iter_mut().for_each(|x| *x += 1);
		}
	    });
	});
    }

    #[bench]
    fn bench_fully_concurrent_update(b: &mut Bencher) {
	let n = 1000;
	let m = 100;
	let mut store: Vec<Mutex<Vec<usize>>> = vec![];
	for _ in 0..m {
	    // create same length of vector (499500)
	    store.push(Mutex::new(vec![0; n * (n - 1) / 2]));
	}
	
        b.iter(|| {
	    (0..m).collect::<Vec<isize>>().par_iter()
		.for_each(|&i| {
		    let mut target = store[i as usize].lock().unwrap();
		    target.iter_mut().for_each(|x| *x += 1);
		});
	});
    }
}
