//! Union-Find on 3D junction boxes.
//!
//! Connect the N closest pairs of junction boxes and find the product
//! of the three largest circuit sizes.

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};
use std::mem;

use anyhow::{anyhow, bail, Context, Result};

/// A point in 3D space.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct Point3 {
    pub x: i64,
    pub y: i64,
    pub z: i64,
}

impl Point3 {
    /// Returns the coordinate along the given axis.
    pub fn component(&self, axis: Axis) -> i64 {
        match axis {
            Axis::X => self.x,
            Axis::Y => self.y,
            Axis::Z => self.z,
        }
    }
}

/// Squared Euclidean distance. True distance requires sqrt, but since sqrt is
/// monotonic, squared distance preserves ordering and avoids the computation.
pub fn distance(target: &Point3, origin: &Point3) -> u64 {
    ((target.x - origin.x).strict_pow(2)
        + (target.y - origin.y).strict_pow(2)
        + (target.z - origin.z).strict_pow(2)) as u64
}

/// Computes squared distance along axis with direction (target relative to origin).
pub fn distance_along_axis(target: &Point3, origin: &Point3, axis: Axis) -> AxisDistance {
    let diff = target.component(axis) - origin.component(axis);
    AxisDistance {
        squared: diff.strict_pow(2) as u64,
        direction: diff.cmp(&0),
    }
}

/// Squared distance along an axis with direction information.
pub struct AxisDistance {
    /// The squared distance.
    pub squared: u64,
    /// Whether target is greater, less, or equal to origin along this axis.
    pub direction: Ordering,
}

/// Axis for KD-tree partitioning, cycling X → Y → Z.
#[derive(Debug, Clone, Copy)]
pub enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    /// Returns the next axis in the cycle.
    pub fn next(&self) -> Self {
        match self {
            Axis::X => Axis::Y,
            Axis::Y => Axis::Z,
            Axis::Z => Axis::X,
        }
    }
}

/// 3D KD-tree for efficient nearest-neighbor queries.
pub struct KDTree<'a> {
    /// The median point at this node.
    pub root: &'a Point3,
    /// The axis used for partitioning at this level.
    pub axis: Axis,
    /// Points with smaller coordinate along `axis`.
    pub left: Option<Box<KDTree<'a>>>,
    /// Points with larger coordinate along `axis`.
    pub right: Option<Box<KDTree<'a>>>,
}

/// Pre-order depth-first iterator over KD-tree points.
pub struct KDTreeIter<'a> {
    stack: Vec<&'a KDTree<'a>>,
}

impl<'a> Iterator for KDTreeIter<'a> {
    type Item = &'a Point3;

    fn next(&mut self) -> Option<Self::Item> {
        let node = self.stack.pop()?;
        // Push right first so left is popped first (pre-order: root, left, right)
        if let Some(right) = &node.right {
            self.stack.push(right);
        }
        if let Some(left) = &node.left {
            self.stack.push(left);
        }
        Some(node.root)
    }
}

impl<'a> KDTree<'a> {
    /// Builds a KD-tree from points. Reorders the input slice during construction.
    pub fn build(points: &'a mut [Point3]) -> Option<Self> {
        Self::build_internal(points, Axis::X)
    }

    /// Pre-order depth-first traversal: root, then left subtree, then right subtree.
    pub fn iter(&self) -> KDTreeIter<'_> {
        KDTreeIter { stack: vec![self] }
    }

    /// Minimum squared distance from point to left/right subtree regions.
    pub fn min_dist_left_right(&self, point: &Point3) -> (u64, u64) {
        // If a point falls into a region, we can't tighten the lower bound to the nearest point.
        // If a point does not fall into a region, it is at least as far away from the nearest point
        // in that region as the shortest distance to that region (normal axis).
        let axis_dist = distance_along_axis(point, self.root, self.axis);
        match axis_dist.direction {
            Ordering::Greater => (axis_dist.squared, 0),
            Ordering::Less => (0, axis_dist.squared),
            Ordering::Equal => (0, 0),
        }
    }

    fn build_internal(points: &'a mut [Point3], axis: Axis) -> Option<KDTree<'a>> {
        if points.is_empty() {
            return None;
        }

        let (left, root, right) = Self::split_along_axis(points, axis);

        Some(Self {
            root,
            axis,
            left: KDTree::build_internal(left, axis.next()).map(Box::new),
            right: KDTree::build_internal(right, axis.next()).map(Box::new),
        })
    }

    fn split_along_axis(
        points: &mut [Point3],
        axis: Axis,
    ) -> (&mut [Point3], &mut Point3, &mut [Point3]) {
        let mid = points.len() / 2;
        points.select_nth_unstable_by_key(mid, |point| point.component(axis))
    }
}

/// Parse input into list of 3D coordinates.
pub fn parse_input(input: &str) -> Result<Vec<Point3>> {
    input
        .lines()
        .map(|line| {
            let coords = line
                .split(',')
                .map(|coord| coord.parse::<i64>().context("invalid coordinate"))
                .collect::<Result<Vec<_>>>()?;

            let &[x, y, z] = coords.as_slice() else {
                bail!("expected 3 coordinates");
            };
            Ok(Point3 { x, y, z })
        })
        .collect()
}

/// Connect the `connections` closest pairs and return the product of the 3 largest circuit sizes.
pub fn solve_part1(
    junctions: &mut [Point3],
    connections: usize,
    num_of_networks: usize,
) -> Option<u64> {
    let kd_tree = KDTree::build(junctions)?;

    let junctions = kd_tree.iter().cloned().collect::<Vec<_>>();
    // Reverse wraps NNPriorityQueue to create a min-heap: we want the globally
    // closest pair, so the queue with smallest next-neighbor distance pops first.
    let mut heap = BinaryHeap::from_iter(
        junctions
            .iter()
            .map(|junction| Reverse(NNPriorityQueue::new(junction, &kd_tree))),
    );

    let mut networks = DisjointSetForest::new(&junctions);

    let mut connection_count = 0;
    while connection_count < connections {
        let Reverse(mut nn_queue) = heap.pop()?;
        let junction = nn_queue.query_point();
        match nn_queue.pop() {
            None => continue,
            Some(other_junction) => {
                heap.push(Reverse(nn_queue));
                // on another queue, other_junction and junction are exactly reversed,
                // so we need to skip one of the pairs to avoid double counting an existing connection
                if junction > other_junction {
                    continue;
                }
                connection_count += 1;
                networks.connect(junction, other_junction).ok()?;
            }
        }
    }

    networks
        .sizes_descending()
        .take(num_of_networks)
        .try_fold(1u64, |acc, size| acc.checked_mul(size))
}

struct DisjointSetNode {
    pub parent: usize,
    pub size: u64,
}

/// Union-Find structure for tracking connected components (circuits).
struct DisjointSetForest {
    sets: Vec<DisjointSetNode>,
    index: HashMap<Point3, usize>,
}

impl DisjointSetForest {
    pub fn new(junctions: &[Point3]) -> Self {
        let mut forest = Self {
            sets: Vec::with_capacity(junctions.len()),
            index: HashMap::with_capacity(junctions.len()),
        };

        for (i, junction) in junctions.iter().enumerate() {
            forest.sets.push(DisjointSetNode { parent: i, size: 1 });
            forest.index.insert(*junction, i);
        }

        forest
    }

    pub fn connect(&mut self, junction1: &Point3, junction2: &Point3) -> Result<()> {
        let index1 = *self
            .index
            .get(junction1)
            .ok_or(anyhow!("Missing junction"))?;
        let index2 = *self
            .index
            .get(junction2)
            .ok_or(anyhow!("Missing junction"))?;

        self.merge(index1, index2);

        Ok(())
    }

    pub fn sizes_descending(&self) -> impl Iterator<Item = u64> {
        let mut heap: BinaryHeap<u64> = self
            .sets
            .iter()
            .filter_map(|set| (set.size > 0).then_some(set.size))
            .collect();
        std::iter::from_fn(move || heap.pop())
    }

    fn merge(&mut self, index1: usize, index2: usize) {
        let root1 = self.find_root(index1);
        let root2 = self.find_root(index2);

        // Deterministic parent selection: smaller index becomes root.
        // Ensures consistent merging regardless of argument order.
        let (root1, root2) = match root1.cmp(&root2) {
            Ordering::Equal => {
                // sets are already connected
                return;
            }
            Ordering::Less => (root1, root2),
            Ordering::Greater => (root2, root1),
        };

        let DisjointSetNode { size: size2, .. } = mem::replace(
            &mut self.sets[root2],
            DisjointSetNode {
                parent: root1,
                size: 0,
            },
        );

        // because we already found the roots of the merged sets, we do not need to recurse
        // and can be sure the sets have been merged successfully
        self.sets[root1].size += size2;
    }

    /// Path compression: flatten the tree as we traverse, so future lookups are O(1).
    fn find_root(&mut self, index: usize) -> usize {
        let parent = self.sets[index].parent;
        if parent == index {
            index
        } else {
            let root = self.find_root(parent);
            self.sets[index].parent = root;
            root
        }
    }
}

/// Lazily yields nearest neighbors from a KD-tree, ordered by increasing distance.
struct NNPriorityQueue<'a> {
    heap: BinaryHeap<Reverse<HeapNode<'a>>>,
    point: &'a Point3,
    /// Cached next point and its squared distance. Ensures comparisons reflect
    /// actual neighbor distances rather than internal heap state (SubTree bounds).
    next: Option<(&'a Point3, u64)>,
}

impl<'a> NNPriorityQueue<'a> {
    pub fn new(point: &'a Point3, kd_tree: &'a KDTree<'a>) -> Self {
        let mut queue = Self {
            heap: BinaryHeap::new(),
            point,
            next: None,
        };
        queue.heap.push(Reverse(HeapNode::subtree(kd_tree, 0)));
        queue.next = queue.find_next_point();
        queue
    }

    /// Returns the query point (the point we're finding neighbors for).
    pub fn query_point(&self) -> &'a Point3 {
        self.point
    }

    pub fn pop(&mut self) -> Option<&'a Point3> {
        let (coords, _) = self.next.take()?;
        self.next = self.find_next_point();
        Some(coords)
    }

    fn find_next_point(&mut self) -> Option<(&'a Point3, u64)> {
        let Reverse(node) = self.heap.pop()?;

        match node {
            HeapNode::Point { coords, .. } if coords == self.point => {
                self.find_next_point() // skip self
            }
            HeapNode::Point { coords, distance } => Some((coords, distance)),
            HeapNode::SubTree { tree, .. } => {
                self.heap.push(Reverse(HeapNode::point(
                    tree.root,
                    distance(tree.root, self.point),
                )));
                let (left_dist, right_dist) = tree.min_dist_left_right(self.point);
                if let Some(left) = &tree.left {
                    self.heap.push(Reverse(HeapNode::subtree(left, left_dist)));
                }
                if let Some(right) = &tree.right {
                    self.heap
                        .push(Reverse(HeapNode::subtree(right, right_dist)));
                }
                self.find_next_point()
            }
        }
    }
}

impl Eq for NNPriorityQueue<'_> {}

impl PartialEq for NNPriorityQueue<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.next.map(|(_, dist)| dist) == other.next.map(|(_, dist)| dist)
    }
}

impl PartialOrd for NNPriorityQueue<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for NNPriorityQueue<'_> {
    /// Compares queues by their next-neighbor distance (larger = Greater).
    ///
    /// Empty queues are treated as "largest" so that when multiple queues are
    /// managed in a min-heap (via `Reverse<NNPriorityQueue>`), exhausted queues
    /// sink to the bottom. Min-heap is the natural choice for nearest-neighbor
    /// problems where you want the globally closest pair, not the farthest.
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.next, other.next) {
            (None, None) => Ordering::Equal,
            (None, Some(_)) => Ordering::Greater,
            (Some(_), None) => Ordering::Less,
            (Some((_, self_dist)), Some((_, other_dist))) => self_dist.cmp(&other_dist),
        }
    }
}

enum HeapNode<'a> {
    Point { coords: &'a Point3, distance: u64 },
    SubTree { tree: &'a KDTree<'a>, distance: u64 },
}

impl<'a> HeapNode<'a> {
    pub fn point(coords: &'a Point3, distance: u64) -> Self {
        Self::Point { coords, distance }
    }

    pub fn subtree(tree: &'a KDTree<'a>, distance: u64) -> Self {
        Self::SubTree { tree, distance }
    }

    pub fn distance(&self) -> u64 {
        match self {
            HeapNode::Point { distance, .. } => *distance,
            HeapNode::SubTree { distance, .. } => *distance,
        }
    }
}

impl Eq for HeapNode<'_> {}

impl PartialEq<Self> for HeapNode<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.distance().eq(&other.distance())
    }
}

impl PartialOrd<Self> for HeapNode<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapNode<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance().cmp(&other.distance())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXAMPLE: &str = "\
162,817,812
57,618,57
906,360,560
592,479,940
352,342,300
466,668,158
542,29,236
431,825,988
739,650,466
52,470,668
216,146,977
819,987,18
117,168,530
805,96,715
346,949,466
970,615,88
941,993,340
862,61,35
984,92,344
425,690,689";

    #[test]
    fn part1() {
        let mut junctions = parse_input(EXAMPLE).unwrap();
        assert_eq!(solve_part1(junctions.as_mut_slice(), 10, 3), Some(40));
    }

    #[test]
    fn point3_component() {
        let point = Point3 { x: 1, y: 2, z: 3 };
        assert_eq!(point.component(Axis::X), 1);
        assert_eq!(point.component(Axis::Y), 2);
        assert_eq!(point.component(Axis::Z), 3);
    }

    #[test]
    fn point3_negative_coordinates() {
        let point = Point3 {
            x: -5,
            y: -10,
            z: -15,
        };
        assert_eq!(point.component(Axis::X), -5);
        assert_eq!(point.component(Axis::Y), -10);
        assert_eq!(point.component(Axis::Z), -15);
    }

    #[test]
    fn axis_next_cycles() {
        assert!(matches!(Axis::X.next(), Axis::Y));
        assert!(matches!(Axis::Y.next(), Axis::Z));
        assert!(matches!(Axis::Z.next(), Axis::X));
    }

    #[test]
    fn distance_same_point() {
        let point = Point3 { x: 5, y: 10, z: 15 };
        assert_eq!(distance(&point, &point), 0);
    }

    #[test]
    fn distance_along_single_axis() {
        let a = Point3 { x: 0, y: 0, z: 0 };
        let b = Point3 { x: 3, y: 0, z: 0 };
        assert_eq!(distance(&a, &b), 9); // 3^2 = 9
    }

    #[test]
    fn distance_3d() {
        let a = Point3 { x: 0, y: 0, z: 0 };
        let b = Point3 { x: 1, y: 2, z: 2 };
        assert_eq!(distance(&a, &b), 9); // 1 + 4 + 4 = 9
    }

    #[test]
    fn distance_is_symmetric() {
        let a = Point3 { x: 1, y: 2, z: 3 };
        let b = Point3 { x: 4, y: 5, z: 6 };
        assert_eq!(distance(&a, &b), distance(&b, &a));
    }

    #[test]
    fn distance_along_axis_positive_direction() {
        let a = Point3 { x: 10, y: 0, z: 0 };
        let b = Point3 { x: 3, y: 0, z: 0 };
        let result = distance_along_axis(&a, &b, Axis::X);
        assert_eq!(result.squared, 49); // (10-3)^2 = 49
        assert_eq!(result.direction, Ordering::Greater);
    }

    #[test]
    fn distance_along_axis_negative_direction() {
        let a = Point3 { x: 3, y: 0, z: 0 };
        let b = Point3 { x: 10, y: 0, z: 0 };
        let result = distance_along_axis(&a, &b, Axis::X);
        assert_eq!(result.squared, 49);
        assert_eq!(result.direction, Ordering::Less);
    }

    #[test]
    fn distance_along_axis_equal() {
        let a = Point3 { x: 5, y: 10, z: 15 };
        let b = Point3 { x: 5, y: 20, z: 25 };
        let result = distance_along_axis(&a, &b, Axis::X);
        assert_eq!(result.squared, 0);
        assert_eq!(result.direction, Ordering::Equal);
    }

    #[test]
    fn distance_along_axis_y() {
        let a = Point3 { x: 0, y: 10, z: 0 };
        let b = Point3 { x: 0, y: 3, z: 0 };
        let result = distance_along_axis(&a, &b, Axis::Y);
        assert_eq!(result.squared, 49);
        assert_eq!(result.direction, Ordering::Greater);
    }

    #[test]
    fn distance_along_axis_z() {
        let a = Point3 { x: 0, y: 0, z: 2 };
        let b = Point3 { x: 0, y: 0, z: 7 };
        let result = distance_along_axis(&a, &b, Axis::Z);
        assert_eq!(result.squared, 25);
        assert_eq!(result.direction, Ordering::Less);
    }

    #[test]
    fn kd_tree_build_empty() {
        let mut points: Vec<Point3> = vec![];
        assert!(KDTree::build(&mut points).is_none());
    }

    #[test]
    fn kd_tree_build_single_point() {
        let mut points = vec![Point3 { x: 1, y: 2, z: 3 }];
        let tree = KDTree::build(&mut points).unwrap();
        assert_eq!(*tree.root, Point3 { x: 1, y: 2, z: 3 });
        assert!(tree.left.is_none());
        assert!(tree.right.is_none());
    }

    #[test]
    fn kd_tree_iter_visits_all_points() {
        let mut points = vec![
            Point3 { x: 5, y: 5, z: 5 },
            Point3 { x: 2, y: 2, z: 2 },
            Point3 { x: 8, y: 8, z: 8 },
            Point3 { x: 1, y: 1, z: 1 },
            Point3 { x: 9, y: 9, z: 9 },
        ];
        let tree = KDTree::build(&mut points).unwrap();
        let visited: Vec<_> = tree.iter().cloned().collect();
        assert_eq!(visited.len(), 5);
    }

    #[test]
    fn kd_tree_min_dist_point_on_left() {
        let mut points = vec![
            Point3 { x: 5, y: 0, z: 0 },
            Point3 { x: 2, y: 0, z: 0 },
            Point3 { x: 8, y: 0, z: 0 },
        ];
        let tree = KDTree::build(&mut points).unwrap();

        // Query point at x=1, which is to the left of root (x=5)
        let query = Point3 { x: 1, y: 0, z: 0 };
        let (left_dist, right_dist) = tree.min_dist_left_right(&query);

        // Point is in left region, so left_dist = 0
        // Distance to right region = (5-1)^2 = 16
        assert_eq!(left_dist, 0);
        assert_eq!(right_dist, 16);
    }

    #[test]
    fn kd_tree_min_dist_point_on_right() {
        let mut points = vec![
            Point3 { x: 5, y: 0, z: 0 },
            Point3 { x: 2, y: 0, z: 0 },
            Point3 { x: 8, y: 0, z: 0 },
        ];
        let tree = KDTree::build(&mut points).unwrap();

        // Query point at x=7, which is to the right of root (x=5)
        let query = Point3 { x: 7, y: 0, z: 0 };
        let (left_dist, right_dist) = tree.min_dist_left_right(&query);

        // Point is in right region, so right_dist = 0
        // Distance to left region = (7-5)^2 = 4
        assert_eq!(left_dist, 4);
        assert_eq!(right_dist, 0);
    }

    #[test]
    fn nn_queue_returns_neighbors_in_distance_order() {
        let mut points = vec![
            Point3 { x: 0, y: 0, z: 0 },
            Point3 { x: 1, y: 0, z: 0 }, // distance 1 from origin
            Point3 { x: 3, y: 0, z: 0 }, // distance 9 from origin
            Point3 { x: 2, y: 0, z: 0 }, // distance 4 from origin
        ];
        let tree = KDTree::build(&mut points).unwrap();

        let origin = Point3 { x: 0, y: 0, z: 0 };
        let mut queue = NNPriorityQueue::new(&origin, &tree);

        // Should return neighbors in order of increasing squared distance
        let first = queue.pop().unwrap();
        assert_eq!(*first, Point3 { x: 1, y: 0, z: 0 });

        let second = queue.pop().unwrap();
        assert_eq!(*second, Point3 { x: 2, y: 0, z: 0 });

        let third = queue.pop().unwrap();
        assert_eq!(*third, Point3 { x: 3, y: 0, z: 0 });

        // No more neighbors (origin itself was skipped)
        assert!(queue.pop().is_none());
    }

    #[test]
    fn nn_queue_skips_self() {
        let mut points = vec![Point3 { x: 5, y: 5, z: 5 }, Point3 { x: 0, y: 0, z: 0 }];
        let tree = KDTree::build(&mut points).unwrap();

        // Query from a point that's in the tree (get it via tree iterator)
        let query = tree.iter().next().unwrap();
        let mut queue = NNPriorityQueue::new(query, &tree);

        // Should skip self and return the other point
        let neighbor = queue.pop().unwrap();
        assert_ne!(*neighbor, *query);

        // Only one neighbor, queue should be empty
        assert!(queue.pop().is_none());
    }

    #[test]
    fn forest_initial_sizes() {
        let points = vec![
            Point3 { x: 0, y: 0, z: 0 },
            Point3 { x: 1, y: 1, z: 1 },
            Point3 { x: 2, y: 2, z: 2 },
        ];
        let forest = DisjointSetForest::new(&points);

        // Initially each point is its own set of size 1, descending order
        let sizes: Vec<_> = forest.sizes_descending().collect();
        assert_eq!(sizes, vec![1, 1, 1]);
    }

    #[test]
    fn forest_merge_two_sets() {
        let points = vec![
            Point3 { x: 0, y: 0, z: 0 },
            Point3 { x: 1, y: 1, z: 1 },
            Point3 { x: 2, y: 2, z: 2 },
        ];
        let mut forest = DisjointSetForest::new(&points);

        forest.connect(&points[0], &points[1]).unwrap();

        // Expect descending order: [2, 1]
        let sizes: Vec<_> = forest.sizes_descending().collect();
        assert_eq!(sizes, vec![2, 1]);
    }

    #[test]
    fn forest_merge_all_sets() {
        let points = vec![
            Point3 { x: 0, y: 0, z: 0 },
            Point3 { x: 1, y: 1, z: 1 },
            Point3 { x: 2, y: 2, z: 2 },
        ];
        let mut forest = DisjointSetForest::new(&points);

        forest.connect(&points[0], &points[1]).unwrap();
        forest.connect(&points[1], &points[2]).unwrap();

        let sizes: Vec<_> = forest.sizes_descending().collect();
        assert_eq!(sizes, vec![3]);
    }

    #[test]
    fn forest_merge_already_connected() {
        let points = vec![Point3 { x: 0, y: 0, z: 0 }, Point3 { x: 1, y: 1, z: 1 }];
        let mut forest = DisjointSetForest::new(&points);

        forest.connect(&points[0], &points[1]).unwrap();
        forest.connect(&points[0], &points[1]).unwrap(); // merge again

        let sizes: Vec<_> = forest.sizes_descending().collect();
        assert_eq!(sizes, vec![2]);
    }

    #[test]
    fn forest_missing_junction_error() {
        let points = vec![Point3 { x: 0, y: 0, z: 0 }];
        let mut forest = DisjointSetForest::new(&points);

        let unknown = Point3 {
            x: 99,
            y: 99,
            z: 99,
        };
        assert!(forest.connect(&points[0], &unknown).is_err());
    }

    #[test]
    fn heap_node_ordering_larger_distance_is_greater() {
        let p1 = Point3 { x: 0, y: 0, z: 0 };
        let p2 = Point3 { x: 1, y: 1, z: 1 };

        let node_small = HeapNode::point(&p1, 5);
        let node_large = HeapNode::point(&p2, 10);

        assert_eq!(node_large.cmp(&node_small), Ordering::Greater);
        assert_eq!(node_small.cmp(&node_large), Ordering::Less);
        assert_eq!(node_small.cmp(&node_small), Ordering::Equal);
    }

    #[test]
    fn nn_queue_ordering_larger_next_distance_is_greater() {
        // Create two separate trees with different nearest-neighbor distances
        let mut points1 = vec![
            Point3 { x: 0, y: 0, z: 0 },
            Point3 { x: 1, y: 0, z: 0 }, // distance 1 from origin
        ];
        let mut points2 = vec![
            Point3 {
                x: 100,
                y: 100,
                z: 100,
            },
            Point3 {
                x: 110,
                y: 100,
                z: 100,
            }, // distance 100 from (100,100,100)
        ];

        let tree1 = KDTree::build(&mut points1).unwrap();
        let tree2 = KDTree::build(&mut points2).unwrap();

        let origin = Point3 { x: 0, y: 0, z: 0 };
        let far_point = Point3 {
            x: 100,
            y: 100,
            z: 100,
        };

        let queue_small = NNPriorityQueue::new(&origin, &tree1); // next dist = 1
        let queue_large = NNPriorityQueue::new(&far_point, &tree2); // next dist = 100

        // Queue with larger next-neighbor distance should be Greater
        assert_eq!(queue_large.cmp(&queue_small), Ordering::Greater);
        assert_eq!(queue_small.cmp(&queue_large), Ordering::Less);
    }

    #[test]
    fn nn_queue_in_reverse_heap_pops_smallest_distance() {
        let mut points1 = vec![
            Point3 { x: 0, y: 0, z: 0 },
            Point3 { x: 1, y: 0, z: 0 }, // distance 1 from origin
        ];
        let mut points2 = vec![
            Point3 {
                x: 100,
                y: 100,
                z: 100,
            },
            Point3 {
                x: 110,
                y: 100,
                z: 100,
            }, // distance 100 from (100,100,100)
        ];

        let tree1 = KDTree::build(&mut points1).unwrap();
        let tree2 = KDTree::build(&mut points2).unwrap();

        let origin = Point3 { x: 0, y: 0, z: 0 };
        let far_point = Point3 {
            x: 100,
            y: 100,
            z: 100,
        };

        let queue_small = NNPriorityQueue::new(&origin, &tree1);
        let queue_large = NNPriorityQueue::new(&far_point, &tree2);

        // Reverse heap should pop the queue with smallest next-neighbor distance first
        let mut heap = BinaryHeap::new();
        heap.push(Reverse(queue_large));
        heap.push(Reverse(queue_small));

        let first = heap.pop().unwrap().0;
        assert_eq!(first.query_point(), &origin); // queue_small should be popped first
    }
}
