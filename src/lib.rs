use fnv::FnvHashMap;
use logos::{Lexer, Logos};
use std::ops::{Add, Mul, Sub};

#[derive(Logos, Debug, PartialEq, Copy, Clone)]
enum Token {
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
    #[token = "end"]
    EndBlock,
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
    Subtract,
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

/// Representation of any value in the scripting language, each variant is a data type.
#[derive(Debug, PartialEq, Clone)]
pub enum Value {
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Text(String),
    Array(Vec<Value>),
    Function(Box<Function>),
}

///A scripting language function
#[derive(Debug, PartialEq, Clone)]
pub struct Function {
    arguments: Vec<String>,
    instructions: Vec<Instruction>,
    ret_val: Expression,
}

///Represents a scripting language program
#[derive(Debug, PartialEq, Clone)]
pub struct State {
    variables: FnvHashMap<String, Value>,
}

#[derive(Debug, PartialEq, Clone)]
enum Instruction {
    Assign(String, Expression),
    Goto(usize),
    ConditionalJump(Expression, usize, usize),
}

#[derive(Debug, PartialEq, Clone)]
enum Expression {
    Call(Box<Expression>, Vec<Expression>),
    Var(String),
    Lit(Value),
    Oper(Operator, Box<Expression>, Box<Expression>),
}

#[derive(Debug, PartialEq, Copy, Clone)]
enum Operator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Eq,
    Neq,
}

impl Function {
    fn from_raw(
        arguments: Vec<String>,
        instructions: Vec<Instruction>,
        ret_val: Expression,
    ) -> Self {
        Function {
            arguments,
            instructions,
            ret_val,
        }
    }
    /// Calls self with specified args and specified global state
    pub fn call(&self, args: Vec<Value>, globals: &State) -> Value {
        let mut inner_state = State::new();
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

impl Instruction {
    pub fn execute(&self, state: &mut State, globals: &State) -> Option<usize> {
        match self {
            Instruction::Assign(name, expr) => {
                state.insert(name.to_string(), expr.evaluate(state, globals));
                None
            }
            Instruction::Goto(line) => Some(*line),
            Instruction::ConditionalJump(condition, if_true, otherwise) => {
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

fn parse_expression(lexer: &mut Lexer<Token, &str>) -> Result<Expression, String> {
    let first = lexer.token;
    let mut expr = match first {
        Token::Integer => {
            Expression::Lit(Value::Integer(lexer.slice().parse().expect(lexer.slice())))
        }
        Token::Float => Expression::Lit(Value::Float(lexer.slice().parse().unwrap())),
        Token::Identifier => Expression::Var(lexer.slice().to_string()),
        _ => return Err(lexer.slice().to_string()),
    };
    lexer.advance();
    if lexer.token == Token::StartParen {
        lexer.advance();
        let mut args = vec![];
        while lexer.token != Token::EndParen {
            match parse_expression(lexer) {
                Ok(expression) =>
                    args.push(expression),
                Err(_) => break,
            }
        }
        lexer.advance();
        expr = Expression::Call(Box::new(expr), args);
    }
    let second = lexer.token;
    match second {
        Token::Add => {
            lexer.advance();
            Ok(Expression::Oper(
                Operator::Add,
                Box::new(expr),
                Box::new(parse_expression(lexer)?),
            ))
        }
        Token::Subtract => {
            lexer.advance();
            Ok(Expression::Oper(
                Operator::Subtract,
                Box::new(expr),
                Box::new(parse_expression(lexer)?),
            ))
        }
        Token::Mul => {
            lexer.advance();
            Ok(Expression::Oper(
                Operator::Multiply,
                Box::new(expr),
                Box::new(parse_expression(lexer)?),
            ))
        }
        Token::Eq => {
            lexer.advance();
            Ok(Expression::Oper(
                Operator::Eq,
                Box::new(expr),
                Box::new(parse_expression(lexer)?),
            ))
        }
        Token::Neq => {
            lexer.advance();
            Ok(Expression::Oper(
                Operator::Neq,
                Box::new(expr),
                Box::new(parse_expression(lexer)?),
            ))
        }
        _ => Ok(expr),
    }
}

enum Block {
    Statement(Instruction),
    Block(Vec<Instruction>),
}

fn parse_assignment(name: String, lexer: &mut Lexer<Token, &str>) -> Result<Block, String> {
    lexer.advance();
    if lexer.token != Token::Assign {
        return Err(lexer.slice().to_string());
    };
    lexer.advance();
    let expr = parse_expression(lexer)?;
    Ok(Block::Statement(Instruction::Assign(name, expr)))
}

fn parse_if(lexer: &mut Lexer<Token, &str>, index: usize) -> Result<Block, String> {
    lexer.advance();
    let expr = parse_expression(lexer)?;
    let mut if_block = Vec::new();
    let mut else_block = Vec::new();
    while lexer.token != Token::End {
        if lexer.token == Token::EndBlock {
            lexer.advance();
            break;
        };
        if lexer.token == Token::Else {
            lexer.advance();
            while lexer.token != Token::End {
                if lexer.token == Token::EndBlock {
                    lexer.advance();
                    break;
                };
                append_block(
                    lexer,
                    index + if_block.len() + else_block.len() + 2,
                    &mut else_block,
                )?;
                lexer.advance();
            }
            break;
        };
        append_block(lexer, index + if_block.len() + 1, &mut if_block)?;
        lexer.advance();
    }
    if !else_block.is_empty() {
        if_block.push(Instruction::Goto(
            index + if_block.len() + else_block.len() + 2,
        ));
    }
    let mut output = vec![Instruction::ConditionalJump(
        expr,
        index + 1,
        index + if_block.len() + 1,
    )];
    output.append(&mut if_block);
    output.append(&mut else_block);
    Ok(Block::Block(output))
}

fn parse_while(lexer: &mut Lexer<Token, &str>, index: usize) -> Result<Block, String> {
    lexer.advance();
    let expr = parse_expression(lexer)?;
    let mut block = Vec::new();
    while lexer.token != Token::End {
        if lexer.token == Token::EndBlock {
            lexer.advance();
            break;
        };
        append_block(lexer, index + block.len() + 1, &mut block)?;
        lexer.advance();
    }
    block.push(Instruction::Goto(index));
    let mut output = vec![Instruction::ConditionalJump(
        expr,
        index + 1,
        index + block.len() + 1,
    )];
    output.append(&mut block);
    Ok(Block::Block(output))
}

fn append_block(
    lexer: &mut Lexer<Token, &str>,
    index: usize,
    block: &mut Vec<Instruction>,
) -> Result<(), String> {
    match parse_block(lexer, index)? {
        Block::Statement(state) => {
            block.push(state);
        }
        Block::Block(mut states) => {
            block.append(&mut states);
        }
    }
    Ok(())
}

fn parse_block(lexer: &mut Lexer<Token, &str>, index: usize) -> Result<Block, String> {
    match lexer.token {
        Token::Identifier => parse_assignment(lexer.slice().to_string(), lexer),
        Token::If => parse_if(lexer, index),
        Token::While => parse_while(lexer, index),
        Token::NewLine => {
            lexer.advance();
            parse_block(lexer, index)
        }
        _ => return Err(lexer.slice().to_string()),
    }
}

fn parse_var<'a>(
    name: &str,
    lexer: &mut Lexer<Token, &str>,
    state: &mut State,
) -> Result<(), String> {
    let first = lexer.token;
    if first != Token::Assign {
        return Err(lexer.slice().to_string());
    };
    lexer.advance();
    let expr = parse_expression(lexer)?;
    state.insert(name.to_string(), expr.evaluate(state, &State::new()));
    Ok(())
}

fn collect_arguments(lexer: &mut Lexer<Token, &str>) -> Result<Vec<String>, String> {
    if lexer.token != Token::StartParen {
        return Err(lexer.slice().to_string());
    };
    lexer.advance();
    let mut args = vec![];
    loop {
        match lexer.token {
            Token::EndParen => {
                lexer.advance();
                return Ok(args);
            }
            Token::Identifier => {
                args.push(lexer.slice().to_string());
                lexer.advance();
            }
            _ => return Err(lexer.slice().to_string()),
        }
    }
}

fn parse_function<'a>(lexer: &mut Lexer<Token, &str>, state: &mut State) -> Result<(), String> {
    let name = if lexer.token == Token::Identifier {
        lexer.slice().to_string()
    } else {
        return Err(lexer.slice().to_string());
    };
    lexer.advance();
    let args = collect_arguments(lexer)?;
    while let Token::NewLine = lexer.token {
        lexer.advance();
    }
    let mut statements = vec![];
    let mut last_lexer = lexer.clone();
    while let Ok(block) = parse_block(lexer, statements.len()) {
        match block {
            Block::Statement(statement) => statements.push(statement),
            Block::Block(mut new_statements) => statements.append(&mut new_statements),
        };
        lexer.advance();
        last_lexer = lexer.clone();
    }
    let ret_val = parse_expression(&mut last_lexer)?;
    state.insert(
        name,
        Value::Function(Box::new(Function::from_raw(args, statements, ret_val))),
    );
    Ok(())
}

fn parse_value<'a>(lexer: &mut Lexer<Token, &str>, state: &mut State) -> Result<(), String> {
    let first = lexer.token;
    let name = lexer.slice();
    lexer.advance();
    match first {
        Token::Identifier => parse_var(name, lexer, state),
        Token::Function => parse_function(lexer, state),
        Token::NewLine => Ok(()),
        _ => Err(name.to_string()),
    }
}

impl State {
    /// Creates a new State containing no data.
    pub fn new() -> Self {
        State {
            variables: FnvHashMap::default(),
        }
    }
    /// Parses given string as a script, and returns the corresponding script
    pub fn from_str(script: &str) -> Result<Self, String> {
        let mut state = State::new();
        let mut lexer = Token::lexer(script);
        while let Ok(_) = parse_value(&mut lexer, &mut state) {}
        Ok(state)
    }
    ///Inserts a value into the State with specified key.
    pub fn insert(&mut self, key: String, value: Value) {
        self.variables.insert(key, value);
    }
    ///Returns the corresponding value if key is in the State, and None otherwise
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.variables.get(key)
    }
    ///Returns the result of calling entry_point if it is found in the State, and None otherwise
    pub fn run(&self, entry_point: &str) -> Option<Value> {
        self.variables.get(entry_point).map(|val| match val {
            Value::Function(main) => main.call(vec![], &self),
            val => val.clone(),
        })
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
        Instruction::Assign(String::from("x"), Expression::Lit(Value::Float(4.0)))
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
            Value::Function(Box::new(Function::from_raw(
                vec![],
                vec![Instruction::Assign(
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
    fn call_function_raw() {
        let mut environment = State::new();

        environment.insert(String::from("x"), Value::Float(-2.5));
        environment.insert(
            String::from("f"),
            Value::Function(Box::new(Function::from_raw(
                vec![String::from("y")],
                vec![],
                Expression::Var(String::from("y")),
            ))),
        );
        environment.insert(
            String::from("main"),
            Value::Function(Box::new(Function::from_raw(
                vec![String::from("y")],
                vec![],
                Expression::Call(
                    Box::new(Expression::Var(String::from("f"))),
                    vec![Expression::Var(String::from("x"))],
                ),
            ))),
        );
        assert_eq!(environment.run("main"), Some(Value::Float(-2.5)));
    }

    #[test]
    fn add_numbers() {
        let mut environment = State::new();

        environment.insert(String::from("x"), Value::Integer(-3));
        environment.insert(String::from("y"), Value::Float(2.5));
        let adder = Function::from_raw(
            vec![String::from("y"), String::from("x")],
            vec![Instruction::Assign(
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
        let adder = Function::from_raw(
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
        let subber = Function::from_raw(
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
        let multer = Function::from_raw(
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
    fn fibonacci_manual() {
        let mut by_hand = State::new();
        by_hand.insert(String::from("x"), Value::Integer(7));
        by_hand.insert(
            String::from("main"),
            Value::Function(Box::new(Function::from_raw(
                vec![],
                vec![
                    Instruction::Assign(String::from("y"), Expression::Lit(Value::Integer(1))),
                    Instruction::Assign(String::from("z"), Expression::Lit(Value::Integer(1))),
                    Instruction::ConditionalJump(
                        Expression::Oper(
                            Operator::Neq,
                            Box::new(Expression::Var(String::from("x"))),
                            Box::new(Expression::Lit(Value::Integer(0))),
                        ),
                        3,
                        8,
                    ),
                    Instruction::Assign(
                        String::from("x"),
                        Expression::Oper(
                            Operator::Subtract,
                            Box::new(Expression::Var(String::from("x"))),
                            Box::new(Expression::Lit(Value::Integer(1))),
                        ),
                    ),
                    Instruction::Assign(
                        String::from("temp"),
                        Expression::Oper(
                            Operator::Add,
                            Box::new(Expression::Var(String::from("y"))),
                            Box::new(Expression::Var(String::from("z"))),
                        ),
                    ),
                    Instruction::Assign(String::from("y"), Expression::Var(String::from("z"))),
                    Instruction::Assign(String::from("z"), Expression::Var(String::from("temp"))),
                    Instruction::Goto(2),
                ],
                Expression::Var(String::from("y")),
            ))),
        );
        assert_eq!(by_hand.run("main"), Some(Value::Integer(21)));
        by_hand.insert(String::from("x"), Value::Integer(8));
        assert_eq!(by_hand.run("main"), Some(Value::Integer(34)));
    }

    #[test]
    fn fibonacci() {
        let mut by_hand = State::new();
        by_hand.insert(String::from("x"), Value::Integer(7));
        by_hand.insert(
            String::from("main"),
            Value::Function(Box::new(Function::from_raw(
                vec![],
                vec![
                    Instruction::Assign(String::from("y"), Expression::Lit(Value::Integer(1))),
                    Instruction::Assign(String::from("z"), Expression::Lit(Value::Integer(1))),
                    Instruction::ConditionalJump(
                        Expression::Oper(
                            Operator::Neq,
                            Box::new(Expression::Var(String::from("x"))),
                            Box::new(Expression::Lit(Value::Integer(0))),
                        ),
                        3,
                        8,
                    ),
                    Instruction::Assign(
                        String::from("x"),
                        Expression::Oper(
                            Operator::Subtract,
                            Box::new(Expression::Var(String::from("x"))),
                            Box::new(Expression::Lit(Value::Integer(1))),
                        ),
                    ),
                    Instruction::Assign(
                        String::from("temp"),
                        Expression::Oper(
                            Operator::Add,
                            Box::new(Expression::Var(String::from("y"))),
                            Box::new(Expression::Var(String::from("z"))),
                        ),
                    ),
                    Instruction::Assign(String::from("y"), Expression::Var(String::from("z"))),
                    Instruction::Assign(String::from("z"), Expression::Var(String::from("temp"))),
                    Instruction::Goto(2),
                ],
                Expression::Var(String::from("y")),
            ))),
        );
        let program = State::from_str(
            "\
x = 7

fn main()
    y = 1
    z = 1
    while x != 0
        x = x - 1
        temp = y + z
        y = z
        z = temp
    end
    y
",
        );
        assert_eq!(program, Ok(by_hand));
    }

    #[test]
    fn nested_while() {
        let program = State::from_str(
            "\
x = 4

fn main()
    y = 3
    z = 3
    while x != 0
        x = x - 1
        while y != 0
            y = y - 1
            z = z + y
        end
        y = z
    end
    z
",
        );
        let program = program.unwrap();
        assert_eq!(program.run("main"), Some(Value::Integer(26_796)));
    }

    #[test]
    fn test_if() {
        let program = State::from_str(
            "\
x = 4

fn main()
    y = 0
    z = 0
    while x != 0
        y = y + 1
        if x == y
             y = 0
        else
            if x == 2
                y = 5
            end
            z = z + x
        end
        x = x - 1
    end
    x + y + z
",
        );
        let program = program.unwrap();
        assert_eq!(program.run("main"), Some(Value::Integer(16)));
    }

    #[test]
    fn call_function() {
        let program = State::from_str(
            "\
fn fib(x)
    if x == 1
        z = 1
    else
        if x == 2
            z = 1
        else
            z = fib(x - 1) + fib(x - 2)
        end
    end
    z

fn main ()
    y = 10
    fib(y)
",
        );
        let program = program.unwrap();
        assert_eq!(program.run("main"), Some(Value::Integer(55)));
    }
}
