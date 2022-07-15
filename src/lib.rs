#![feature(test)]
extern crate test;

use parking_lot::{Mutex, RwLock, RwLockWriteGuard, RwLockReadGuard};
use std::ops::{Deref, DerefMut};

/// Vector Map Builder
///
pub struct VecMapBuilder<T> {
    data: Mutex<Vec<T>>,
    partitions: Mutex<Vec<usize>>,
}

impl<T: Clone> VecMapBuilder<T> {
    
    /// Create new builder.
    pub fn new() -> Self {
	let data = Mutex::new(Vec::new());
	let partitions = Mutex::new(vec![0]);
	Self { data, partitions }
    }

    /// Insert vector.
    /// Returns index of inserted vector.
    pub fn insert(&self, xs: &[T]) -> usize {
	let mut partitions = self.partitions.lock();
	let len = partitions.len();
	let offset = partitions[len - 1];
	partitions.push(offset + xs.len());
	self.data.lock().extend_from_slice(xs);
	len - 1
    }

    /// Build mutable vecmap.
    /// # Examples
    ///
    /// ```
    /// use vecmap::VecMapBuilder;
    /// 
    /// let builder = VecMapBuilder::new();
    /// builder.insert(&[1, 2, 3]);
    /// builder.insert(&[4, 5]);
    /// let vecmap = builder.build_mut();
    /// vecmap.borrow_mut(0)[0] = 2;
    /// assert_eq!([2, 2, 3], *vecmap.borrow_mut(0));
    /// ```
    pub fn build_mut<'a>(&'a self) -> MutVecMap<'a, T> {
	let partitions = self.partitions.lock().iter()
	    .map(|x| RwLock::new(*x))
	    .collect::<Vec<_>>();
	let data = unsafe {
	    let data = self.data.lock();
	    let ptr = data.as_ptr() as *const T;
	    std::slice::from_raw_parts(ptr, data.len())
	};
	MutVecMap {data, partitions}
    }

    /// Build immutable vecmap
    /// # Examples
    ///
    /// ```
    /// use vecmap::VecMapBuilder;
    /// 
    /// let builder = VecMapBuilder::new();
    /// builder.insert(&[1, 2, 3]);
    /// builder.insert(&[4, 5]);
    /// let vecmap = builder.build();
    /// assert_eq!([1, 2, 3], *vecmap.borrow(0));
    /// ```
    pub fn build<'a>(&'a self) -> VecMap<'a, T> {
	let partitions = self.partitions.lock().iter()
	    .map(|x| *x)
	    .collect::<Vec<_>>();
	let data = unsafe {
	    let data = self.data.lock();
	    let ptr = data.as_ptr() as *const T;
	    std::slice::from_raw_parts(ptr, data.len())
	};
	VecMap {data, partitions}
    }
}

/// (Immutable) Vector Mapping
///
/// Immutable version of Mutable Vector Map.
/// 
pub struct VecMap<'a, T> {
    data: &'a [T],
    partitions: Vec<usize>,
}

impl<'a, T: Clone + Sized> VecMap<'a, T> {
    #[inline]
    pub fn borrow(&self, idx: usize) -> &[T] {
	let offset = self.partitions[idx];
	let size = self.partitions[idx + 1] - offset;
	let slice = unsafe {
	    let ptr = self.data.as_ptr().offset(offset as isize) as *mut T;
	    std::slice::from_raw_parts(ptr, size)
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
/// which requires 48 bytes of memory (Arc: 8, parking_lot::RwLock: 16, Vec: 24)
/// in addition to the vector contents.
///
/// For VecMap, parking_lot::RwLock: 16, isize: 8 for a total of 34Byte.
pub struct MutVecMap<'a, T> {
    data: &'a [T],
    partitions: Vec<RwLock<usize>>,
}

impl<'a, T: Clone + Sized> MutVecMap<'a, T> {
    /// borrow returns immutable slice of chunk.
    /// Wrapperd by SliceReadGuard.
    ///
    /// # Example
    ///
    /// ```
    /// use vecmap::VecMapBuilder;
    /// 
    /// let builder = VecMapBuilder::new();
    /// builder.insert(&[1, 2, 3]);
    /// builder.insert(&[4, 5]);
    /// let vecmap = builder.build_mut();
    /// assert_eq!(3, vecmap.borrow(0).len());
    /// ```
    pub fn borrow(&self, idx: usize) -> SliceReadGuard<T> {
	let guard = self.partitions[idx].read();
	let offset = *guard;
	let size = *self.partitions[idx + 1].read() - offset;
	let slice = unsafe {
	    let ptr = self.data.as_ptr().offset(offset as isize);
	    std::slice::from_raw_parts(ptr, size)
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
    /// builder.insert(&[1, 2, 3]);
    /// builder.insert(&[4, 5]);
    /// let vecmap = builder.build_mut();
    /// vecmap.borrow_mut(1)[1] = 9;
    /// assert_eq!([4, 9], *vecmap.borrow(1));
    /// ```
    pub fn borrow_mut(&self, idx: usize) -> SliceWriteGuard<T> {
	let guard = self.partitions[idx].write();
	let offset = *guard;
	let size = *self.partitions[idx + 1].read() - offset;
	let slice = unsafe {
	    let ptr = self.data.as_ptr().offset(offset as isize) as *mut T;
	    std::slice::from_raw_parts_mut(ptr, size as usize)
	};
	SliceWriteGuard { guard, slice }
    }
}

#[derive(Debug)]
pub struct SliceReadGuard<'a, T> {
    guard: RwLockReadGuard<'a, usize>,
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
    guard: RwLockWriteGuard<'a, usize>,
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


unsafe impl<'a, T> Send for MutVecMap<'a, T> {}
unsafe impl<'a, T> Sync for MutVecMap<'a, T> {}
unsafe impl<'a, T> Send for VecMap<'a, T> {}
unsafe impl<'a, T> Sync for VecMap<'a, T> {}


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
	let builder = VecMapBuilder::new();
	for x in 0..n { builder.insert(&vec![1; x]); };
	let vm = builder.build();
	
	(0..m).collect::<Vec<_>>().par_iter()
	    .for_each(|_| {
		for idx in 0..n {
		    let slice = vm.borrow(idx);
		    assert!(slice.iter().all(|&x| x == 1));
		}
	    });
    }

    #[bench]
    fn bench_concurrent_read(b: &mut Bencher) {
	let n = 1000;
	let m = 10;
	let builder = VecMapBuilder::new();
	for x in 0..n { builder.insert(&vec![1; x]); };
	let vm = builder.build();

	let mut rng = rand::thread_rng();
	let xss = (0..m).map(|_| {
	    let mut xs = (0..n).collect::<Vec<_>>();
	    xs.shuffle(&mut rng);
	    xs
	}).collect::<Vec<_>>();
	
        b.iter(|| {
	    xss.par_iter().for_each(|xs| {
		for &idx in xs.iter() {
		    let slice = vm.borrow(idx);
		    slice.iter().fold(0, |x, y| x + y);
		}
	    });
	});
    }

    #[bench]
    fn bench_serialized_read(b: &mut Bencher) {
	let n = 1000;
	let m = 10;
	let builder = VecMapBuilder::new();
	for x in 0..n { builder.insert(&vec![1; x]); };
	let vm = builder.build();

	let mut rng = rand::thread_rng();
	let xss = (0..m).map(|_| {
	    let mut xs = (0..n).collect::<Vec<_>>();
	    xs.shuffle(&mut rng);
	    xs
	}).collect::<Vec<_>>();
	
        b.iter(|| {
	    xss.iter().for_each(|xs| {
		for &idx in xs.iter() {
		    let slice = vm.borrow(idx);
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
	let builder = VecMapBuilder::new();
	let indices = (0..n).into_par_iter()
	    .map(|x| builder.insert(&vec![1; x]));
	assert_eq!((0..n).sum::<usize>(), indices.sum());
	
	let vm = builder.build_mut();
	assert_eq!(n * (n - 1) / 2, vm.data.len());
	assert_eq!(n + 1, vm.partitions.len());
    }

    #[test]
    fn update_concurrent() {
	let n = 1000;
	let m = 100;
	let builder = VecMapBuilder::new();
	let _indices = (0..n).into_par_iter()
	    .for_each(|x| { builder.insert(&vec![1; x]); });

	let vm = builder.build_mut();

	(0..m).collect::<Vec<_>>().par_iter()
	    .for_each(|_| {
		for idx in 0..n {
		    let mut slice = vm.borrow_mut(idx);
		    slice.iter_mut().for_each(|x| *x += 1);
		}
	    });
	// all values +1 in 'm' times.
	assert!(vm.data.iter().all(|&x| x == m + 1));
    }

    #[test]
    fn read_concurrent() {
	let n = 1000;
	let m = 100;
	let builder = VecMapBuilder::new();
	let _indices = (0..n).into_par_iter()
	    .for_each(|x| { builder.insert(&vec![1; x]); });

	let vm = builder.build_mut();

	(0..m).collect::<Vec<_>>().par_iter()
	    .for_each(|_| {
		for idx in 0..n {
		    let slice = vm.borrow(idx);
		    assert!(slice.iter().all(|&x| x == 1));
		}
	    });
    }
  
    #[bench]
    fn bench_concurrent_insert(b: &mut Bencher) {
	let n = 10000;
        b.iter(|| {
	    let builder = VecMapBuilder::new();
	    (0..n).collect::<Vec<_>>().into_par_iter()
		.for_each(|x| { builder.insert(&vec![1; x]); });
	});
    }    

    #[bench]
    fn bench_serialized_insert(b: &mut Bencher) {
	let n = 10000;
        b.iter(|| {
	    let builder = VecMapBuilder::new();
	    (0..n).collect::<Vec<_>>().into_iter()
		.for_each(|x| { builder.insert(&vec![1; x]); });
	});
    }
    
    #[bench]
    fn bench_concurrent_update(b: &mut Bencher) {
	let n = 1000;
	let m = 100;
	let builder = VecMapBuilder::new();
	for x in 0..n { builder.insert(&vec![1; x]); };

	let vm = builder.build_mut();

	let mut rng = rand::thread_rng();
	let xss = (0..m).map(|_| {
	    let mut xs = (0..n).collect::<Vec<_>>();
	    xs.shuffle(&mut rng);
	    xs
	}).collect::<Vec<_>>();
	
        b.iter(|| {
	    xss.par_iter().for_each(|xs| {
		for &idx in xs.iter() {
		    let mut slice = vm.borrow_mut(idx);
		    slice.iter_mut().for_each(|x| *x += 1);
		}
	    });
	});
    }

    #[bench]
    fn bench_concurrent_read(b: &mut Bencher) {
	let n = 1000;
	let m = 100;
	let builder = VecMapBuilder::new();
	for x in 0..n { builder.insert(&vec![1; x]); };

	let vm = builder.build_mut();

	let mut rng = rand::thread_rng();
	let xss = (0..m).map(|_| {
	    let mut xs = (0..n).collect::<Vec<_>>();
	    xs.shuffle(&mut rng);
	    xs
	}).collect::<Vec<_>>();
	
        b.iter(|| {
	    xss.par_iter().for_each(|xs| {
		for &idx in xs.iter() {
		    let slice = vm.borrow(idx);
		    slice.iter().fold(0, |x, y| x + y);
		}
	    });
	});
    }

    #[bench]
    fn bench_serialized_update(b: &mut Bencher) {
	let n = 1000;
	let m = 100;
	let builder = VecMapBuilder::new();
	for x in 0..n { builder.insert(&vec![1; x]); };

	let vm = builder.build_mut();

	let mut rng = rand::thread_rng();
	let xss = (0..m).map(|_| {
	    let mut xs = (0..n).collect::<Vec<_>>();
	    xs.shuffle(&mut rng);
	    xs
	}).collect::<Vec<_>>();
	
        b.iter(|| {
	    xss.iter().for_each(|xs| {
		for &idx in xs.iter() {
		    let mut slice = vm.borrow_mut(idx);
		    slice.iter_mut().for_each(|x| *x += 1);
		}
	    });
	});
    }

    #[bench]
    fn bench_for_compare_fully_concurrent_update(b: &mut Bencher) {
	let n = 1000;
	let m = 100;
	let mut store: Vec<Mutex<Vec<usize>>> = vec![];
	for _ in 0..m {
	    store.push(Mutex::new(vec![0; n * (n - 1) / 2]));
	}
	
        b.iter(|| {
	    (0..m).collect::<Vec<isize>>().par_iter()
		.for_each(|&i| {
		    let mut target = store[i as usize].lock();
		    target.iter_mut().for_each(|x| *x += 1);
		});
	});
    }
}
