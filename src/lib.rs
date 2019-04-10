use std::collections::HashMap;

#[derive(Debug, PartialEq, Clone)]
pub enum Value {
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Text(String),
    Array(Vec<Value>),
}

pub struct Function {
    instructions: Vec<Instruction>,
    ret_val: Expression,
}

pub struct ProgramState {
    variables: HashMap<String, Value>,
}

pub enum Instruction {
    Assign(String, Expression),
}

pub enum Expression {
    Var(String),
    Lit(Value),
}

impl Function {
    pub fn from(instructions: Vec<Instruction>, ret_val: Expression) -> Self {
        Function {
            instructions,
            ret_val
        }
    }
    pub fn call(&self, state: &mut ProgramState) -> Value {
        for instruction in self.instructions.iter() {
            instruction.execute(state)
        };
        self.ret_val.evaluate(state)
    }
}

impl Instruction {
    pub fn execute(&self, state: &mut ProgramState) {
        match self {
            Instruction::Assign(name, expr) => state.insert(name.to_string(), expr.evaluate(state)),
        }
    }
}

impl Expression {
    pub fn evaluate(&self, state: &ProgramState) -> Value {
        match self {
            Expression::Var(id) => state.get(id).expect("Syntax Error").clone(),
            Expression::Lit(val) => val.clone(),
        }
    }
}

impl ProgramState {
    pub fn new() -> Self {
        ProgramState {
            variables: HashMap::new(),
        }
    }
    pub fn insert(&mut self, key: String, value: Value) {
        self.variables.insert(key, value);
    }
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.variables.get(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_variable() {
        let mut environment = ProgramState::new();
        environment.insert(String::from("x"), Value::Integer(3));
        assert_eq!(environment.get("x"), Some(&Value::Integer(3)));
        assert_eq!(environment.get("y"), None);
    }

    #[test]
    fn execute_instruction() {
        let mut environment = ProgramState::new();
        Instruction::Assign(String::from("x"), Expression::Lit(Value::Float(4.0)))
            .execute(&mut environment);
        assert_eq!(environment.get("x"), Some(&Value::Float(4.0)));
        assert_eq!(environment.get("y"), None);
    }

    #[test]
    fn call_function() {
        let mut environment = ProgramState::new();
        let function = Function::from(vec![Instruction::Assign(String::from("x"), Expression::Lit(Value::Float(4.0)))],Expression::Var(String::from("x")));
        assert_eq!(function.call(&mut environment), Value::Float(4.0));
    }
}
