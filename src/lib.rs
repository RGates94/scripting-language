use std::collections::HashMap;
use std::ops::Add;

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
    arguments: Vec<String>,
    instructions: Vec<Statement>,
    ret_val: Expression,
}

#[derive(Debug, PartialEq, Clone)]
pub struct State {
    variables: HashMap<String, Value>,
}

#[derive(Debug, PartialEq, Clone)]
pub enum Statement {
    Assign(String, Expression),
}

#[derive(Debug, PartialEq, Clone)]
pub enum Expression {
    Call(Box<Expression>, Vec<Expression>),
    Var(String),
    Lit(Value),
    Oper(Operator, Box<Expression>, Box<Expression>),
}

#[derive(Debug, PartialEq, Clone)]
pub enum Operator {
    Add,
    Subtract,
    Multiply,
    Divide,
}

impl Function {
    pub fn from(arguments: Vec<String>, instructions: Vec<Statement>, ret_val: Expression) -> Self {
        Function {
            arguments,
            instructions,
            ret_val,
        }
    }
    pub fn call(&self, args: Vec<Value>, globals: &State) -> Value {
        let mut inner_state = State::new();
        for (name, value) in self.arguments.iter().zip(args) {
            inner_state.insert(name.clone(), value)
        }
        for instruction in self.instructions.iter() {
            instruction.execute(&mut inner_state, globals)
        }
        self.ret_val.evaluate(&inner_state, globals)
    }
}

impl Statement {
    pub fn execute(&self, state: &mut State, globals: &State) {
        match self {
            Statement::Assign(name, expr) => {
                state.insert(name.to_string(), expr.evaluate(state, globals))
            }
        }
    }
}

impl Expression {
    pub fn evaluate(&self, state: &State, globals: &State) -> Value {
        match self {
            Expression::Call(name, args) => match name.evaluate(state, globals) {
                Value::Function(inner) => inner.call(
                    args.iter()
                        .map(|expr| expr.evaluate(state, globals))
                        .collect(),
                    globals,
                ),
                _ => panic!("Tried to call a non-function value"),
            },
            Expression::Var(id) => state
                .get(id)
                .or(globals.get(id))
                .expect("Variable not found")
                .clone(),
            Expression::Lit(val) => val.clone(),
            Expression::Oper(op_code, left, right) => match op_code {
                Operator::Add => (left.evaluate(state, globals) + right.evaluate(state, globals))
                    .expect("Failed to add"),
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

impl State {
    pub fn new() -> Self {
        State {
            variables: HashMap::new(),
        }
    }
    pub fn insert(&mut self, key: String, value: Value) {
        self.variables.insert(key, value);
    }
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.variables.get(key)
    }
    pub fn run(&self) -> Value {
        match self.variables.get("main") {
            Some(Value::Function(main)) => main.call(vec![], &self),
            Some(val) => val.clone(),
            None => panic!("No main found"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_variable() {
        let mut environment = State::new();
        environment.insert(String::from("x"), Value::Integer(3));
        assert_eq!(environment.get("x"), Some(&Value::Integer(3)));
        assert_eq!(environment.get("y"), None);
    }

    #[test]
    fn execute_instruction() {
        let mut environment = State::new();
        Statement::Assign(String::from("x"), Expression::Lit(Value::Float(4.0)))
            .execute(&mut environment, &State::new());
        assert_eq!(environment.get("x"), Some(&Value::Float(4.0)));
        assert_eq!(environment.get("y"), None);
    }

    #[test]
    fn evaluate_expression() {
        let mut environment = State::new();
        environment.insert(String::from("x"), Value::Integer(-3));
        environment.insert(
            String::from("f"),
            Value::Function(Box::new(Function::from(
                vec![],
                vec![Statement::Assign(
                    String::from("x"),
                    Expression::Lit(Value::Text(String::from("Hi!"))),
                )],
                Expression::Var(String::from("x")),
            ))),
        );
        assert_eq!(
            Expression::Lit(Value::Float(4.0)).evaluate(&environment, &State::new()),
            Value::Float(4.0)
        );
        assert_eq!(
            Expression::Var(String::from("x")).evaluate(&environment, &State::new()),
            Value::Integer(-3)
        );
        assert_eq!(
            Expression::Call(Box::new(Expression::Var(String::from("f"))), vec![])
                .evaluate(&environment, &State::new()),
            Value::Text(String::from("Hi!"))
        )
    }

    #[test]
    fn call_function() {
        let mut environment = State::new();

        environment.insert(String::from("x"), Value::Float(-2.5));
        environment.insert(
            String::from("f"),
            Value::Function(Box::new(Function::from(
                vec![String::from("y")],
                vec![],
                Expression::Var(String::from("y")),
            ))),
        );
        environment.insert(
            String::from("main"),
            Value::Function(Box::new(Function::from(
                vec![String::from("y")],
                vec![],
                Expression::Call(
                    Box::new(Expression::Var(String::from("f"))),
                    vec![Expression::Var(String::from("x"))],
                ),
            ))),
        );
        assert_eq!(environment.run(), Value::Float(-2.5));
    }

    #[test]
    fn add_numbers() {
        let mut environment = State::new();

        environment.insert(String::from("x"), Value::Integer(-3));
        environment.insert(String::from("y"), Value::Float(2.5));
        let adder = Function::from(
            vec![String::from("y"), String::from("x")],
            vec![Statement::Assign(
                String::from("z"),
                Expression::Oper(
                    Operator::Add,
                    Box::new(Expression::Var(String::from("x"))),
                    Box::new(Expression::Var(String::from("y"))),
                ),
            )],
            Expression::Oper(
                Operator::Add,
                Box::new(Expression::Var(String::from("z"))),
                Box::new(Expression::Var(String::from("x"))),
            ),
        );
        assert_eq!(
            adder.call(
                vec![
                    Expression::Var(String::from("x")).evaluate(&environment, &State::new()),
                    Expression::Var(String::from("x")).evaluate(&environment, &State::new())
                ],
                &State::new()
            ),
            Value::Integer(-9)
        );
        assert_eq!(
            adder.call(
                vec![
                    Expression::Var(String::from("x")).evaluate(&environment, &State::new()),
                    Expression::Var(String::from("y")).evaluate(&environment, &State::new())
                ],
                &State::new()
            ),
            Value::Float(2.0)
        );
        assert_eq!(
            adder.call(
                vec![
                    Expression::Var(String::from("y")).evaluate(&environment, &State::new()),
                    Expression::Var(String::from("x")).evaluate(&environment, &State::new())
                ],
                &State::new()
            ),
            Value::Float(-3.5)
        );
        assert_eq!(
            adder.call(
                vec![
                    Expression::Var(String::from("y")).evaluate(&environment, &State::new()),
                    Expression::Var(String::from("y")).evaluate(&environment, &State::new())
                ],
                &State::new()
            ),
            Value::Float(7.5)
        );
    }

    #[test]
    fn add_text() {
        let mut environment = State::new();

        environment.insert(String::from("x"), Value::Integer(-3));
        environment.insert(String::from("y"), Value::Float(2.5));
        environment.insert(String::from("h"), Value::Text(String::from("Hello, ")));
        environment.insert(String::from("w"), Value::Text(String::from("World!")));
        let adder = Function::from(
            vec![String::from("x"), String::from("y")],
            vec![],
            Expression::Oper(
                Operator::Add,
                Box::new(Expression::Var(String::from("x"))),
                Box::new(Expression::Var(String::from("y"))),
            ),
        );
        assert_eq!(
            adder.call(
                vec![
                    Expression::Var(String::from("h")).evaluate(&environment, &State::new()),
                    Expression::Var(String::from("w")).evaluate(&environment, &State::new())
                ],
                &State::new()
            ),
            Value::Text(String::from("Hello, World!"))
        );
        assert_eq!(
            adder.call(
                vec![
                    Expression::Var(String::from("h")).evaluate(&environment, &State::new()),
                    Expression::Var(String::from("x")).evaluate(&environment, &State::new())
                ],
                &State::new()
            ),
            Value::Text(String::from("Hello, -3"))
        );
        assert_eq!(
            adder.call(
                vec![
                    Expression::Var(String::from("h")).evaluate(&environment, &State::new()),
                    Expression::Var(String::from("y")).evaluate(&environment, &State::new())
                ],
                &State::new()
            ),
            Value::Text(String::from("Hello, 2.5"))
        );
        assert_eq!(
            adder.call(
                vec![
                    Expression::Var(String::from("x")).evaluate(&environment, &State::new()),
                    Expression::Var(String::from("w")).evaluate(&environment, &State::new())
                ],
                &State::new()
            ),
            Value::Text(String::from("-3World!"))
        );
        assert_eq!(
            adder.call(
                vec![
                    Expression::Var(String::from("y")).evaluate(&environment, &State::new()),
                    Expression::Var(String::from("w")).evaluate(&environment, &State::new())
                ],
                &State::new()
            ),
            Value::Text(String::from("2.5World!"))
        );
    }
}
