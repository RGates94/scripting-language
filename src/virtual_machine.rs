use crate::ast::Operator;
use std::ops::{Add, Mul, Sub};

//The structs in this module currently overlap with most of the functionality from ast.rs
//Many of these will likely be merged in the future

/// Representation of any value in the scripting language, each variant is a data type.
#[derive(Debug, PartialEq, Clone)]
pub enum Value {
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Text(String),
    Array(Vec<Value>),
    Function(Box<CompiledFunction>),
}

///A compiled scripting language function
#[derive(Debug, PartialEq, Clone)]
pub struct CompiledFunction {
    arguments: usize,
    capacity: usize,
    instructions: Vec<LinearizedInstruction>,
}

///Represents a scripting language program
#[derive(Debug, PartialEq, Clone)]
pub struct Program {
    variables: Vec<Value>,
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) enum LinearizedInstruction {
    Assign(usize, LinearizedExpression),
    AssignLiteral(usize, Value),
    CopyRelative(usize, usize),
    Add(usize, usize, usize),
    Subtract(usize, usize, usize),
    Eq(usize, usize, usize),
    Neq(usize, usize, usize),
    Goto(usize),
    ConditionalJump(usize, usize, usize),
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) enum LinearizedExpression {
    Call(Box<LinearizedExpression>, Vec<LinearizedExpression>),
    Var(usize),
    Lit(Value),
    Oper(
        Operator,
        Box<LinearizedExpression>,
        Box<LinearizedExpression>,
    ),
}

impl CompiledFunction {
    pub(crate) fn from_raw(
        arguments: usize,
        capacity: usize,
        instructions: Vec<LinearizedInstruction>,
    ) -> Self {
        CompiledFunction {
            arguments,
            capacity,
            instructions,
        }
    }
    /// Calls self with specified args and specified global state
    pub fn call(&self, args: Vec<Value>, program: &mut Program, local_address: usize) -> Value {
        (0..self.arguments)
            .zip(args)
            .for_each(|(name, value)| program.insert(name + local_address, value));
        let mut current_instruction = 0;
        while let Some(instruction) = self.instructions.get(current_instruction) {
            if let Some(line) =
                instruction.execute(program, local_address, local_address + self.capacity)
            {
                current_instruction = line;
            } else {
                current_instruction += 1;
            }
        }
        program.variables[local_address].clone()
    }
}

impl LinearizedInstruction {
    pub fn execute(
        &self,
        program: &mut Program,
        local_address: usize,
        next_frame: usize,
    ) -> Option<usize> {
        match self {
            LinearizedInstruction::Assign(name, expr) => {
                let ret = expr.evaluate(program, local_address, next_frame);
                program.insert(*name + local_address, ret);
                None
            }
            LinearizedInstruction::AssignLiteral(name, val) => {
                program.insert(*name + local_address, val.clone());
                None
            }
            LinearizedInstruction::CopyRelative(name, var_address) => {
                program.copy(*name + local_address, var_address + local_address);
                None
            }
            LinearizedInstruction::Add(name, left, right) => {
                program.add(
                    name + local_address,
                    left + local_address,
                    right + local_address,
                );
                None
            }
            LinearizedInstruction::Subtract(name, left, right) => {
                program.subtract(
                    name + local_address,
                    left + local_address,
                    right + local_address,
                );
                None
            }
            LinearizedInstruction::Eq(name, left, right) => {
                program.equal(
                    name + local_address,
                    left + local_address,
                    right + local_address,
                );
                None
            }
            LinearizedInstruction::Neq(name, left, right) => {
                program.n_equal(
                    name + local_address,
                    left + local_address,
                    right + local_address,
                );
                None
            }
            LinearizedInstruction::Goto(line) => Some(*line),
            LinearizedInstruction::ConditionalJump(condition, if_true, otherwise) => {
                if program.variables[*condition + local_address] == Value::Boolean(true) {
                    Some(*if_true)
                } else {
                    Some(*otherwise)
                }
            }
        }
    }
}

impl LinearizedExpression {
    pub fn evaluate(
        &self,
        program: &mut Program,
        local_address: usize,
        next_frame: usize,
    ) -> Value {
        match self {
            LinearizedExpression::Call(name, args) => {
                match name.evaluate(program, local_address, next_frame) {
                    Value::Function(inner) => inner.call(
                        args.iter()
                            .map(|expr| expr.evaluate(program, local_address, next_frame))
                            .collect(),
                        program,
                        next_frame,
                    ),
                    _ => panic!("Tried to call a non-function value"),
                }
            }
            LinearizedExpression::Var(id) => if *id + local_address < next_frame {
                program.get(*id + local_address)
            } else {
                program.get(*id)
            }
            .expect("Variable not found")
            .clone(),
            LinearizedExpression::Lit(val) => val.clone(),
            LinearizedExpression::Oper(op_code, left, right) => match op_code {
                Operator::Add => (left.evaluate(program, local_address, next_frame)
                    + right.evaluate(program, local_address, next_frame))
                .expect("Failed to add"),
                Operator::Subtract => (left.evaluate(program, local_address, next_frame)
                    - right.evaluate(program, local_address, next_frame))
                .expect("Failed to add"),
                Operator::Multiply => (left.evaluate(program, local_address, next_frame)
                    * right.evaluate(program, local_address, next_frame))
                .expect("Failed to add"),
                Operator::Eq => Value::Boolean(
                    left.evaluate(program, local_address, next_frame)
                        == right.evaluate(program, local_address, next_frame),
                ),
                Operator::Neq => Value::Boolean(
                    left.evaluate(program, local_address, next_frame)
                        != right.evaluate(program, local_address, next_frame),
                ),
                _ => panic!("Not implemented"),
            },
        }
    }
}

impl Add for Value {
    type Output = Option<Value>;
    fn add(self, rhs: Value) -> Option<Value> {
        match self {
            Value::Integer(lhs) => match rhs {
                Value::Integer(rhs) => Some(Value::Integer(lhs + rhs)),
                Value::Float(rhs) => Some(Value::Float(lhs as f64 + rhs)),
                Value::Text(rhs) => Some(Value::Text(format!("{}{}", lhs, rhs))),
                _ => None,
            },
            Value::Float(lhs) => match rhs {
                Value::Integer(rhs) => Some(Value::Float(lhs + rhs as f64)),
                Value::Float(rhs) => Some(Value::Float(lhs + rhs)),
                Value::Text(rhs) => Some(Value::Text(format!("{}{}", lhs, rhs))),
                _ => None,
            },
            Value::Text(lhs) => match rhs {
                Value::Integer(rhs) => Some(Value::Text(format!("{}{}", lhs, rhs))),
                Value::Float(rhs) => Some(Value::Text(format!("{}{}", lhs, rhs))),
                Value::Text(rhs) => Some(Value::Text(format!("{}{}", lhs, rhs))),
                _ => None,
            },
            _ => None,
        }
    }
}

impl Sub for Value {
    type Output = Option<Value>;
    fn sub(self, rhs: Value) -> Option<Value> {
        match self {
            Value::Integer(lhs) => match rhs {
                Value::Integer(rhs) => Some(Value::Integer(lhs - rhs)),
                Value::Float(rhs) => Some(Value::Float(lhs as f64 - rhs)),
                _ => None,
            },
            Value::Float(lhs) => match rhs {
                Value::Integer(rhs) => Some(Value::Float(lhs - rhs as f64)),
                Value::Float(rhs) => Some(Value::Float(lhs - rhs)),
                _ => None,
            },
            _ => None,
        }
    }
}

impl Mul for Value {
    type Output = Option<Value>;
    fn mul(self, rhs: Value) -> Option<Value> {
        match self {
            Value::Integer(lhs) => match rhs {
                Value::Integer(rhs) => Some(Value::Integer(lhs * rhs)),
                Value::Float(rhs) => Some(Value::Float(lhs as f64 * rhs)),
                _ => None,
            },
            Value::Float(lhs) => match rhs {
                Value::Integer(rhs) => Some(Value::Float(lhs * rhs as f64)),
                Value::Float(rhs) => Some(Value::Float(lhs * rhs)),
                _ => None,
            },
            _ => None,
        }
    }
}

impl Program {
    /// Creates a new State containing no data.
    pub fn with_capacity(capacity: usize) -> Self {
        Program {
            variables: vec![Value::Integer(0); capacity],
        }
    }
    pub fn from(variables: Vec<Value>) -> Self {
        Program { variables }
    }
    ///Inserts a value into the State with specified key.
    pub fn insert(&mut self, key: usize, value: Value) {
        while key >= self.variables.len() {
            self.variables.push(Value::Integer(0));
        }
        self.variables[key] = value;
    }
    fn copy(&mut self, destination: usize, origin: usize) {
        if let Some(n) = self.variables.get(origin) {
            self.variables[destination] = n.clone();
        }
    }
    fn add(&mut self, destination: usize, left: usize, right: usize) {
        self.variables[destination] =
            (self.variables[left].clone() + self.variables[right].clone()).expect("failed to add");
    }
    fn subtract(&mut self, destination: usize, left: usize, right: usize) {
        self.variables[destination] = (self.variables[left].clone()
            - self.variables[right].clone())
        .expect("failed to subtract");
    }
    fn equal(&mut self, destination: usize, left: usize, right: usize) {
        self.variables[destination] =
            Value::Boolean(self.variables[left].clone() == self.variables[right].clone())
    }
    fn n_equal(&mut self, destination: usize, left: usize, right: usize) {
        self.variables[destination] =
            Value::Boolean(self.variables[left].clone() != self.variables[right].clone())
    }
    ///Returns the corresponding value if key is in the State, and None otherwise
    pub fn get(&self, key: usize) -> Option<&Value> {
        self.variables.get(key)
    }
    ///Returns the result of calling entry_point if it is found in the State, and None otherwise
    pub fn run(&mut self, entry_point: usize, args: Vec<Value>) -> Option<Value> {
        let n = self.variables.get(entry_point).cloned();
        n.map(|val| match val {
            Value::Function(main) => main.call(args, self, self.variables.len()),
            val => val.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_while_manual() {
        let mut program = Program::from(vec![
            Value::Integer(7),
            Value::Function(Box::new(CompiledFunction::from_raw(
                1,
                6,
                vec![
                    LinearizedInstruction::AssignLiteral(4, Value::Integer(0)),
                    LinearizedInstruction::AssignLiteral(5, Value::Integer(1)),
                    LinearizedInstruction::AssignLiteral(1, Value::Integer(3)),
                    LinearizedInstruction::AssignLiteral(2, Value::Integer(3)),
                    LinearizedInstruction::Neq(3, 0, 4),
                    LinearizedInstruction::ConditionalJump(3, 6, 14),
                    LinearizedInstruction::Subtract(0, 0, 5),
                    LinearizedInstruction::Neq(3, 1, 4),
                    LinearizedInstruction::ConditionalJump(3, 9, 12),
                    LinearizedInstruction::Subtract(1, 1, 5),
                    LinearizedInstruction::Add(2, 1, 2),
                    LinearizedInstruction::Goto(7),
                    LinearizedInstruction::CopyRelative(1, 2),
                    LinearizedInstruction::Goto(4),
                    LinearizedInstruction::CopyRelative(0, 1),
                ],
            ))),
        ]);

        assert_eq!(
            program.run(1, vec![Value::Integer(4)]),
            Some(Value::Integer(26_796))
        );
    }

    #[test]
    fn fibonacci_manual() {
        let mut fib = Program::from(vec![
            Value::Integer(7),
            Value::Integer(7),
            Value::Integer(7),
            Value::Integer(7),
            Value::Integer(7),
            Value::Function(Box::new(CompiledFunction::from_raw(
                1,
                5,
                vec![
                    LinearizedInstruction::AssignLiteral(3, Value::Integer(1)),
                    LinearizedInstruction::AssignLiteral(4, Value::Integer(2)),
                    LinearizedInstruction::Eq(1, 0, 3),
                    LinearizedInstruction::ConditionalJump(1, 4, 6),
                    LinearizedInstruction::AssignLiteral(0, Value::Integer(1)),
                    LinearizedInstruction::Goto(15),
                    LinearizedInstruction::Eq(1, 0, 4),
                    LinearizedInstruction::ConditionalJump(1, 8, 10),
                    LinearizedInstruction::AssignLiteral(0, Value::Integer(1)),
                    LinearizedInstruction::Goto(15),
                    LinearizedInstruction::Subtract(1, 0, 3),
                    LinearizedInstruction::Assign(
                        1,
                        LinearizedExpression::Call(
                            Box::new(LinearizedExpression::Var(5)),
                            vec![LinearizedExpression::Var(1)],
                        ),
                    ),
                    LinearizedInstruction::Subtract(2, 0, 4),
                    LinearizedInstruction::Assign(
                        2,
                        LinearizedExpression::Call(
                            Box::new(LinearizedExpression::Var(5)),
                            vec![LinearizedExpression::Var(2)],
                        ),
                    ),
                    LinearizedInstruction::Add(0, 1, 2),
                ],
            ))),
        ]);
        assert_eq!(
            fib.run(5, vec![Value::Integer(10)]),
            Some(Value::Integer(55))
        );
    }
}
