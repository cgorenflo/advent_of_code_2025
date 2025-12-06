//! Cephalopod math worksheet solver.
//!
//! Input is a grid where math problems (e.g., `123 * 45 * 6`) are arranged in columns,
//! separated by space columns. The operator (+/*) is at the bottom of each column.
//!
//! **Part 1**: Each row within a problem column is a complete number.
//! Transpose the grid so columns become problem vectors, then evaluate.
//!
//! **Part 2**: Each character column is a single digit position. Numbers are
//! written vertically (top=hundreds, bottom=ones). Read columns right-to-left,
//! grouping digit columns into numbers until hitting an operator.

use std::fs;
use std::str::FromStr;

use anyhow::{Context, Result};
use itertools::Itertools;

#[derive(Debug, Clone, Copy)]
pub enum Term {
    Number(u64),
    Operator(Operator),
}

impl Term {
    fn operator(self) -> Option<Operator> {
        match self {
            Term::Operator(o) => Some(o),
            _ => None,
        }
    }

    fn number(self) -> Option<u64> {
        match self {
            Term::Number(n) => Some(n),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Operator {
    Add,
    Multiply,
}

impl Operator {
    fn identity_element(&self) -> u64 {
        match self {
            Operator::Add => 0,
            Operator::Multiply => 1,
        }
    }

    fn apply(&self, op1: u64, op2: u64) -> u64 {
        match self {
            Operator::Add => op1 + op2,
            Operator::Multiply => op1 * op2,
        }
    }
}

impl FromStr for Term {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "+" => Ok(Term::Operator(Operator::Add)),
            "*" => Ok(Term::Operator(Operator::Multiply)),
            v => Ok(Term::Number(v.parse().context("Invalid number")?)),
        }
    }
}

pub fn read_input(path: &str) -> Result<String> {
    Ok(fs::read_to_string(path)?)
}

pub fn solve_worksheet(input: &str, parser: fn(&str) -> Vec<Vec<Term>>) -> u64 {
    parser(input)
        .into_iter()
        .filter_map(|problem| {
            let operator = problem.last()?.operator()?;

            let result = problem
                .iter()
                .filter_map(|term| term.number())
                .fold(operator.identity_element(), |acc, num| operator.apply(acc, num));
            Some(result)
        })
        .sum()
}

pub fn parse_problems_part1(input: &str) -> Vec<Vec<Term>> {
    let term_table: Vec<Vec<Term>> = input
        .lines()
        .map(str::split_whitespace)
        .filter_map(line_to_terms)
        .collect();

    transpose(&term_table)
}

pub fn parse_problems_part2(input: &str) -> Vec<Vec<Term>> {
    let lines: Vec<&str> = input.lines().collect();

    (0..longest_width(&lines))
        .rev()
        .flat_map(move |col| {
            lines
                .iter()
                .filter_map(move |row| row.chars().nth(col))
                .filter(|&char| char != ' ')
                // assert: the line is either a number, or a number followed by an operator
                .chunk_by(|&char| char.is_numeric())
                .into_iter()
                .filter_map(|(_, chars)| chars.collect::<String>().parse::<Term>().ok())
                .collect_vec()
        })
        .collect_vec()
        // assert: operator terms separate problems
        .split_inclusive(|term| term.operator().is_some())
        .map(ToOwned::to_owned)
        .collect()
}

fn line_to_terms<'a>(line: impl Iterator<Item = &'a str>) -> Option<Vec<Term>> {
    line.map(Term::from_str).collect::<Result<_>>().ok()
}

fn transpose<I, T>(term_table: &[I]) -> Vec<Vec<T>>
where
    I: AsRef<[T]>,
    T: Copy,
{
    (0..longest_width(term_table))
        .map(|col| {
            term_table
                .iter()
                .filter_map(|row| row.as_ref().get(col).copied())
                .collect()
        })
        .collect()
}

fn longest_width<I, T>(lines: &[I]) -> usize
where
    I: AsRef<[T]>,
{
    lines
        .iter()
        .map(|terms| terms.as_ref().len())
        .max()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXAMPLE: &str = "\
123 328  51 64
 45 64  387 23
  6 98  215 314
*   +   *   +  ";

    #[test]
    fn part1() {
        assert_eq!(solve_worksheet(EXAMPLE, parse_problems_part1), 4277556);
    }

    #[test]
    fn part2() {
        assert_eq!(solve_worksheet(EXAMPLE, parse_problems_part2), 3263827)
    }
}
