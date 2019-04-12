use logos::Logos;
use std::collections::HashMap;
use std::ops::{Add, Mul, Sub};

pub fn parse_script(script: &str) -> Vec<Token> {
    let mut lexer = PreToken::lexer(script);
    let mut tokens = vec![];

    loop {
        match lexer.token {
            PreToken::Function => {
                lexer.advance();
                match lexer.token {
                    PreToken::Identifier => {
                        tokens.push(Token::Function(String::from(lexer.slice())))
                    }
                    _ => panic!("Expected Identifier"),
                }
            }
            PreToken::If => tokens.push(Token::If),
            PreToken::Else => tokens.push(Token::Else),
            PreToken::For => tokens.push(Token::For),
            PreToken::While => tokens.push(Token::While),
            PreToken::EndIf => tokens.push(Token::EndIf),
            PreToken::EndFor => tokens.push(Token::EndFor),
            PreToken::EndWhile => tokens.push(Token::EndWhile),
            PreToken::StartParen => tokens.push(Token::StartParen),
            PreToken::EndParen => tokens.push(Token::EndParen),
            PreToken::Eq => tokens.push(Token::Oper(Operator::Eq)),
            PreToken::Neq => tokens.push(Token::Oper(Operator::Neq)),
            PreToken::Assign => tokens.push(Token::Assign),
            PreToken::Add => tokens.push(Token::Oper(Operator::Add)),
            PreToken::Sub => tokens.push(Token::Oper(Operator::Subtract)),
            PreToken::Mul => tokens.push(Token::Oper(Operator::Multiply)),
            PreToken::Float => tokens.push(Token::Literal(Value::Float(
                lexer
                    .slice()
                    .parse()
                    .expect("Could not parse float from float token, internal error"),
            ))),
            PreToken::Integer => tokens.push(Token::Literal(Value::Integer(
                lexer
                    .slice()
                    .parse()
                    .expect("Could not parse integer from integer token, internal error"),
            ))),
            PreToken::NewLine => tokens.push(Token::NewLine),
            PreToken::Identifier => tokens.push(Token::Variable(String::from(lexer.slice()))),
            PreToken::End | PreToken::Error => break,
        };
        lexer.advance();
    }
    tokens
}

#[derive(Logos, Debug, PartialEq, Copy, Clone)]
pub enum PreToken {
    #[token = "fn"]
    Function,
    #[token = "if"]
    If,
    #[token = "else"]
    Else,
    #[token = "for"]
    For,
    #[token = "while"]
    While,
    #[token = "end if"]
    EndIf,
    #[token = "end for"]
    EndFor,
    #[token = "end while"]
    EndWhile,
    #[token = "("]
    StartParen,
    #[token = ")"]
    EndParen,
    #[token = "=="]
    Eq,
    #[token = "!="]
    Neq,
    #[token = "="]
    Assign,
    #[token = "+"]
    Add,
    #[token = "-"]
    Sub,
    #[token = "*"]
    Mul,
    #[regex = "[0-9]+[.][0-9]+"]
    Float,
    #[regex = "[0-9]+"]
    Integer,
    #[token = "\n"]
    NewLine,
    #[regex = "[a-zA-Z][a-zA-Z0-9]*"]
    Identifier,
    #[end]
    End,
    #[error]
    Error,
}

#[derive(Debug, PartialEq, Clone)]
pub enum Token {
    Literal(Value),
    Variable(String),
    Function(String),
    Oper(Operator),
    If,
    Else,
    For,
    While,
    EndIf, //EndIf EndFor and EndWhile should go away with more robust handling
    EndFor,
    EndWhile,
    StartParen,
    EndParen,
    Assign,
    NewLine,
}

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
    Goto(usize),
    ConditionalJump(Expression, usize, usize),
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
    Eq,
    Neq,
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

impl Statement {
    pub fn execute(&self, state: &mut State, globals: &State) -> Option<usize> {
        match self {
            Statement::Assign(name, expr) => {
                state.insert(name.to_string(), expr.evaluate(state, globals));
                None
            }
            Statement::Goto(line) => Some(*line),
            Statement::ConditionalJump(condition, if_true, otherwise) => {
                if condition.evaluate(state, globals) == Value::Boolean(true) {
                    Some(*if_true)
                } else {
                    Some(*otherwise)
                }
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

    #[test]
    fn sub_numbers() {
        let mut environment = State::new();

        environment.insert(String::from("x"), Value::Integer(-3));
        environment.insert(String::from("y"), Value::Float(2.5));
        environment.insert(String::from("x1"), Value::Integer(4));
        environment.insert(String::from("y1"), Value::Float(2.25));
        let subber = Function::from(
            vec![String::from("x"), String::from("y")],
            vec![],
            Expression::Oper(
                Operator::Subtract,
                Box::new(Expression::Var(String::from("x"))),
                Box::new(Expression::Var(String::from("y"))),
            ),
        );
        assert_eq!(
            subber.call(
                vec![
                    Expression::Var(String::from("x")).evaluate(&environment, &State::new()),
                    Expression::Var(String::from("x1")).evaluate(&environment, &State::new())
                ],
                &State::new()
            ),
            Value::Integer(-7)
        );
        assert_eq!(
            subber.call(
                vec![
                    Expression::Var(String::from("x")).evaluate(&environment, &State::new()),
                    Expression::Var(String::from("y")).evaluate(&environment, &State::new())
                ],
                &State::new()
            ),
            Value::Float(-5.5)
        );
        assert_eq!(
            subber.call(
                vec![
                    Expression::Var(String::from("y")).evaluate(&environment, &State::new()),
                    Expression::Var(String::from("x")).evaluate(&environment, &State::new())
                ],
                &State::new()
            ),
            Value::Float(5.5)
        );
        assert_eq!(
            subber.call(
                vec![
                    Expression::Var(String::from("y")).evaluate(&environment, &State::new()),
                    Expression::Var(String::from("y1")).evaluate(&environment, &State::new())
                ],
                &State::new()
            ),
            Value::Float(0.25)
        );
    }
    #[test]
    fn mult_numbers() {
        let mut environment = State::new();

        environment.insert(String::from("x"), Value::Integer(-3));
        environment.insert(String::from("y"), Value::Float(2.5));
        environment.insert(String::from("x1"), Value::Integer(-4));
        environment.insert(String::from("y1"), Value::Float(2.25));
        let multer = Function::from(
            vec![String::from("x"), String::from("y")],
            vec![],
            Expression::Oper(
                Operator::Multiply,
                Box::new(Expression::Var(String::from("x"))),
                Box::new(Expression::Var(String::from("y"))),
            ),
        );
        assert_eq!(
            multer.call(
                vec![
                    Expression::Var(String::from("x")).evaluate(&environment, &State::new()),
                    Expression::Var(String::from("x1")).evaluate(&environment, &State::new())
                ],
                &State::new()
            ),
            Value::Integer(12)
        );
        assert_eq!(
            multer.call(
                vec![
                    Expression::Var(String::from("x")).evaluate(&environment, &State::new()),
                    Expression::Var(String::from("y")).evaluate(&environment, &State::new())
                ],
                &State::new()
            ),
            Value::Float(-7.5)
        );
        assert_eq!(
            multer.call(
                vec![
                    Expression::Var(String::from("y")).evaluate(&environment, &State::new()),
                    Expression::Var(String::from("x")).evaluate(&environment, &State::new())
                ],
                &State::new()
            ),
            Value::Float(-7.5)
        );
        assert_eq!(
            multer.call(
                vec![
                    Expression::Var(String::from("y")).evaluate(&environment, &State::new()),
                    Expression::Var(String::from("y1")).evaluate(&environment, &State::new())
                ],
                &State::new()
            ),
            Value::Float(5.625)
        );
    }

    #[test]
    fn fibonacci() {
        let mut environment = State::new();
        environment.insert(String::from("x"), Value::Integer(7));
        environment.insert(
            String::from("main"),
            Value::Function(Box::new(Function::from(
                vec![],
                vec![
                    Statement::Assign(String::from("y"), Expression::Lit(Value::Integer(1))),
                    Statement::Assign(String::from("z"), Expression::Lit(Value::Integer(1))),
                    Statement::ConditionalJump(
                        Expression::Oper(
                            Operator::Neq,
                            Box::new(Expression::Var(String::from("x"))),
                            Box::new(Expression::Lit(Value::Integer(0))),
                        ),
                        3,
                        10,
                    ),
                    Statement::Assign(
                        String::from("x"),
                        Expression::Oper(
                            Operator::Subtract,
                            Box::new(Expression::Var(String::from("x"))),
                            Box::new(Expression::Lit(Value::Integer(1))),
                        ),
                    ),
                    Statement::Assign(
                        String::from("temp"),
                        Expression::Oper(
                            Operator::Add,
                            Box::new(Expression::Var(String::from("y"))),
                            Box::new(Expression::Var(String::from("z"))),
                        ),
                    ),
                    Statement::Assign(String::from("y"), Expression::Var(String::from("z"))),
                    Statement::Assign(String::from("z"), Expression::Var(String::from("temp"))),
                    Statement::Goto(2),
                ],
                Expression::Var(String::from("y")),
            ))),
        );
        assert_eq!(environment.run(), Value::Integer(21));
        environment.insert(String::from("x"), Value::Integer(8));
        assert_eq!(environment.run(), Value::Integer(34));
    }

    #[test]
    fn tokenize() {
        use Token::*;
        let tokens = parse_script(
            "\
x = 7.5

fn main()
    y = 1
    z = 1
    while x != 0
        x = x - 1
        temp = y * z
        y = z
        z = temp
    end while
    y
",
        );
        assert_eq!(
            tokens,
            vec![
                Variable(String::from("x")),
                Assign,
                Literal(Value::Float(7.5)),
                NewLine,
                NewLine,
                Function(String::from("main")),
                StartParen,
                EndParen,
                NewLine,
                Variable(String::from("y")),
                Assign,
                Literal(Value::Integer(1)),
                NewLine,
                Variable(String::from("z")),
                Assign,
                Literal(Value::Integer(1)),
                NewLine,
                While,
                Variable(String::from("x")),
                Oper(Operator::Neq),
                Literal(Value::Integer(0)),
                NewLine,
                Variable(String::from("x")),
                Assign,
                Variable(String::from("x")),
                Oper(Operator::Subtract),
                Literal(Value::Integer(1)),
                NewLine,
                Variable(String::from("temp")),
                Assign,
                Variable(String::from("y")),
                Oper(Operator::Multiply),
                Variable(String::from("z")),
                NewLine,
                Variable(String::from("y")),
                Assign,
                Variable(String::from("z")),
                NewLine,
                Variable(String::from("z")),
                Assign,
                Variable(String::from("temp")),
                NewLine,
                EndWhile,
                NewLine,
                Variable(String::from("y")),
                NewLine
            ]
        )
    }
}
