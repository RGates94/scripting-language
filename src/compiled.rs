use std::ops::{Add, Mul, Sub};
use crate::ast::Operator;

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
    arguments: Vec<usize>,
    capacity: usize,
    instructions: Vec<CompiledInstruction>,
    ret_val: CompiledExpression,
}

///Represents a scripting language program
#[derive(Debug, PartialEq, Clone)]
pub struct Program {
    variables: Vec<Value>,
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) enum CompiledInstruction {
    Assign(usize, CompiledExpression),
    Goto(usize),
    ConditionalJump(CompiledExpression, usize, usize),
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) enum CompiledExpression {
    Call(Box<CompiledExpression>, Vec<CompiledExpression>),
    Var(usize),
    Lit(Value),
    Oper(Operator, Box<CompiledExpression>, Box<CompiledExpression>),
}

impl CompiledFunction {
    pub(crate) fn from_raw(
        arguments: Vec<usize>,
        capacity: usize,
        instructions: Vec<CompiledInstruction>,
        ret_val: CompiledExpression,
    ) -> Self {
        CompiledFunction {
            arguments,
            capacity,
            instructions,
            ret_val,
        }
    }
    /// Calls self with specified args and specified global state
    pub fn call(&self, args: Vec<Value>, globals: &Program) -> Value {
        let mut inner_state = Program::with_capacity(self.capacity);
        self.arguments
            .iter()
            .zip(args)
            .for_each(|(name, value)| inner_state.insert(name.clone(), value));
        let mut current_instruction = 0;
        while let Some(instruction) = self.instructions.get(current_instruction) {
            if let Some(line) = instruction.execute(&mut inner_state, globals) {
                current_instruction = line;
            } else {
                current_instruction += 1;
            }
        }
        self.ret_val.evaluate(&inner_state, globals)
    }
}

impl CompiledInstruction {
    pub fn execute(&self, state: &mut Program, globals: &Program) -> Option<usize> {
        match self {
            CompiledInstruction::Assign(name, expr) => {
                state.insert(*name, expr.evaluate(state, globals));
                None
            }
            CompiledInstruction::Goto(line) => Some(*line),
            CompiledInstruction::ConditionalJump(condition, if_true, otherwise) => {
                if condition.evaluate(state, globals) == Value::Boolean(true) {
                    Some(*if_true)
                } else {
                    Some(*otherwise)
                }
            }
        }
    }
}

impl CompiledExpression {
    pub fn evaluate(&self, state: &Program, globals: &Program) -> Value {
        match self {
            CompiledExpression::Call(name, args) => match name.evaluate(state, globals) {
                Value::Function(inner) => inner.call(
                    args.iter()
                        .map(|expr| expr.evaluate(state, globals))
                        .collect(),
                    globals,
                ),
                _ => panic!("Tried to call a non-function value"),
            },
            CompiledExpression::Var(id) => state
                .get(*id)
                .or(globals.get(*id))
                .expect("Variable not found")
                .clone(),
            CompiledExpression::Lit(val) => val.clone(),
            CompiledExpression::Oper(op_code, left, right) => match op_code {
                Operator::Add => (left.evaluate(state, globals) + right.evaluate(state, globals))
                    .expect("Failed to add"),
                Operator::Subtract => (left.evaluate(state, globals)
                    - right.evaluate(state, globals))
                .expect("Failed to add"),
                Operator::Multiply => (left.evaluate(state, globals)
                    * right.evaluate(state, globals))
                .expect("Failed to add"),
                Operator::Eq => {
                    Value::Boolean(left.evaluate(state, globals) == right.evaluate(state, globals))
                }
                Operator::Neq => {
                    Value::Boolean(left.evaluate(state, globals) != right.evaluate(state, globals))
                }
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
        self.variables[key] = value;
    }
    ///Returns the corresponding value if key is in the State, and None otherwise
    pub fn get(&self, key: usize) -> Option<&Value> {
        self.variables.get(key)
    }
    ///Returns the result of calling entry_point if it is found in the State, and None otherwise
    pub fn run(&self, entry_point: usize, args: Vec<Value>) -> Option<Value> {
        self.variables.get(entry_point).map(|val| match val {
            Value::Function(main) => main.call(args, &self),
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
                vec![0],
                3,
                vec![
                    CompiledInstruction::Assign(1, CompiledExpression::Lit(Value::Integer(3))),
                    CompiledInstruction::Assign(2, CompiledExpression::Lit(Value::Integer(3))),
                    CompiledInstruction::ConditionalJump(
                        CompiledExpression::Oper(
                            Operator::Neq,
                            Box::new(CompiledExpression::Var(0)),
                            Box::new(CompiledExpression::Lit(Value::Integer(0))),
                        ),
                        3,
                        10,
                    ),
                    CompiledInstruction::Assign(
                        0,
                        CompiledExpression::Oper(
                            Operator::Subtract,
                            Box::new(CompiledExpression::Var(0)),
                            Box::new(CompiledExpression::Lit(Value::Integer(1))),
                        ),
                    ),
                    CompiledInstruction::ConditionalJump(
                        CompiledExpression::Oper(
                            Operator::Neq,
                            Box::new(CompiledExpression::Var(1)),
                            Box::new(CompiledExpression::Lit(Value::Integer(0))),
                        ),
                        5,
                        8,
                    ),
                    CompiledInstruction::Assign(
                        1,
                        CompiledExpression::Oper(
                            Operator::Subtract,
                            Box::new(CompiledExpression::Var(1)),
                            Box::new(CompiledExpression::Lit(Value::Integer(1))),
                        ),
                    ),
                    CompiledInstruction::Assign(
                        2,
                        CompiledExpression::Oper(
                            Operator::Add,
                            Box::new(CompiledExpression::Var(1)),
                            Box::new(CompiledExpression::Var(2)),
                        ),
                    ),
                    CompiledInstruction::Goto(4),
                    CompiledInstruction::Assign(1, CompiledExpression::Var(2)),
                    CompiledInstruction::Goto(2),
                ],
                CompiledExpression::Var(1),
            ))),]);

        assert_eq!(program.run(1, vec![Value::Integer(4)]), Some(Value::Integer(26_796)));}

    #[test]
    fn fibonacci_manual() {

        let fib = Program::from(vec![
            Value::Integer(7),
            Value::Integer(7),
            Value::Integer(7),
            Value::Function(Box::new(CompiledFunction::from_raw(
                vec![0],
                2,
                vec![
                    CompiledInstruction::ConditionalJump(
                        CompiledExpression::Oper(
                            Operator::Eq,
                            Box::new(CompiledExpression::Var(0)),
                            Box::new(CompiledExpression::Lit(Value::Integer(1))),
                        ),
                        1,
                        3,
                    ),
                    CompiledInstruction::Assign(
                        1,
                        CompiledExpression::Lit(Value::Integer(1),
                        ),
                    ),
                    CompiledInstruction::Goto(7),
                    CompiledInstruction::ConditionalJump(
                        CompiledExpression::Oper(
                            Operator::Eq,
                            Box::new(CompiledExpression::Var(0)),
                            Box::new(CompiledExpression::Lit(Value::Integer(2))),
                        ),
                        4,
                        6,
                    ),
                    CompiledInstruction::Assign(
                        1,
                        CompiledExpression::Lit(Value::Integer(1)),
                    ),
                    CompiledInstruction::Goto(7),
                    CompiledInstruction::Assign(
                        1,
                        CompiledExpression::Oper(
                            Operator::Add,
                            Box::new(CompiledExpression::Call(Box::new(CompiledExpression::Var(3)), vec![CompiledExpression::Oper(Operator::Subtract, Box::new(CompiledExpression::Var(0)), Box::new(CompiledExpression::Lit(Value::Integer(1))))])),
                            Box::new(CompiledExpression::Call(Box::new(CompiledExpression::Var(3)), vec![CompiledExpression::Oper(Operator::Subtract, Box::new(CompiledExpression::Var(0)), Box::new(CompiledExpression::Lit(Value::Integer(2))))])),
                        ),
                    ),
                ],
                CompiledExpression::Var(1),
            ))),]);
        assert_eq!(fib.run(3, vec![Value::Integer(10)]), Some(Value::Integer(55)));

    }
}