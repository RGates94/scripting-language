use std::collections::HashMap;

#[derive(Debug, PartialEq, Clone)]
pub enum Value {
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Text(String),
    Array(Vec<Value>),
    Function(Box<Function>),
}

#[derive(Debug, PartialEq, Clone)]
pub struct Function {
    instructions: Vec<Instruction>,
    ret_val: Expression,
}

#[derive(Debug, PartialEq, Clone)]
pub struct ProgramState {
    variables: HashMap<String, Value>,
}

#[derive(Debug, PartialEq, Clone)]
pub enum Instruction {
    Assign(String, Expression),
}

#[derive(Debug, PartialEq, Clone)]
pub enum Expression {
    Call(String),
    Var(String),
    Lit(Value),
}

impl Function {
    pub fn from(instructions: Vec<Instruction>, ret_val: Expression) -> Self {
        Function {
            instructions,
            ret_val,
        }
    }
    pub fn call(&self) -> Value {
        let mut inner_state = ProgramState::new();
        for instruction in self.instructions.iter() {
            instruction.execute(&mut inner_state)
        }
        self.ret_val.evaluate(&mut inner_state)
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
            Expression::Call(name) => match state.get(name).expect("Syntax Error") {
                Value::Function(inner) => inner.call(),
                _ => panic!("Tried to call a non-function value"),
            },
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
    fn evaluate_expression() {
        let mut environment = ProgramState::new();
        environment.insert(String::from("x"), Value::Integer(-3));
        environment.insert(
            String::from("f"),
            Value::Function(Box::new(Function::from(
                vec![Instruction::Assign(
                    String::from("x"),
                    Expression::Lit(Value::Text(String::from("Hi!"))),
                )],
                Expression::Var(String::from("x")),
            ))),
        );
        assert_eq!(
            Expression::Lit(Value::Float(4.0)).evaluate(&environment),
            Value::Float(4.0)
        );
        assert_eq!(
            Expression::Var(String::from("x")).evaluate(&environment),
            Value::Integer(-3)
        );
        assert_eq!(
            Expression::Call(String::from("f")).evaluate(&environment),
            Value::Text(String::from("Hi!"))
        )
    }

    #[test]
    fn call_function() {
        let mut environment = ProgramState::new();
        let function = Function::from(
            vec![Instruction::Assign(
                String::from("x"),
                Expression::Lit(Value::Float(4.0)),
            )],
            Expression::Var(String::from("x")),
        );
        assert_eq!(function.call(), Value::Float(4.0));
    }
}
