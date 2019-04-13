use logos::{Lexer, Logos};
use std::collections::HashMap;
use std::ops::{Add, Mul, Sub};

fn pre_tokens_to_tokens(lexer: &mut Lexer<PreToken, &str>) -> Vec<Token> {
    let mut tokens = vec![];

    loop {
        match lexer.token {
            PreToken::Function => {
                lexer.advance();
                if lexer.token == PreToken::Identifier {
                    tokens.push(Token::Function(String::from(lexer.slice())))
                } else {
                    panic!("Expected Identifier")
                }
            }
            PreToken::If => tokens.push(Token::If),
            PreToken::Else => tokens.push(Token::Else),
            PreToken::For => tokens.push(Token::For),
            PreToken::While => {
                lexer.advance();
                tokens.push(Token::While(pre_tokens_to_tokens(lexer)));
                if lexer.token == PreToken::EndBlock {
                    lexer.advance()
                } else {
                    break;
                }
            }
            PreToken::StartParen => tokens.push(Token::StartParen),
            PreToken::EndParen => tokens.push(Token::EndParen),
            PreToken::Eq => tokens.push(Token::Eq),
            PreToken::Neq => tokens.push(Token::Neq),
            PreToken::Assign => tokens.push(Token::Assign),
            PreToken::Add => tokens.push(Token::Add),
            PreToken::Subtract => tokens.push(Token::Subtract),
            PreToken::Mul => tokens.push(Token::Multiply),
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
            _ => break,
        };
        lexer.advance();
    }
    tokens
}

#[derive(Logos, Debug, PartialEq, Copy, Clone)]
enum PreToken {
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

#[derive(Debug, PartialEq, Clone)]
enum Token {
    Literal(Value),
    Variable(String),
    Function(String),
    Add,
    Subtract,
    Multiply,
    Eq,
    Neq,
    If,
    Else,
    For,
    While(Vec<Token>),
    EndIf, //EndIf and EndFor should go away with more robust handling
    EndFor,
    StartParen,
    EndParen,
    Assign,
    NewLine,
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
    variables: HashMap<String, Value>,
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

fn parse_expression(tokens: &[Token]) -> Option<(Expression, &[Token])> {
    let (first, tokens) = match tokens.split_first() {
        Some((first, tokens)) => (first, tokens),
        None => return None,
    };
    let expr = match first {
        Token::Literal(value) => Expression::Lit(value.clone()),
        Token::Variable(name) => Expression::Var(name.clone()),
        _ => return None,
    };
    let (second, tokens) = match tokens.split_first() {
        Some((first, tokens)) => (first, tokens),
        None => return Some((expr, tokens)),
    };
    match second {
        Token::NewLine => Some((expr, tokens)),
        Token::Add => match parse_expression(tokens) {
            Some((right, tokens)) => Some((
                Expression::Oper(Operator::Add, Box::new(expr), Box::new(right)),
                tokens,
            )),
            _ => None,
        }
        Token::Subtract => match parse_expression(tokens) {
            Some((right, tokens)) => Some((
                Expression::Oper(Operator::Subtract, Box::new(expr), Box::new(right)),
                tokens,
            )),
            _ => None,
        }
        Token::Multiply => match parse_expression(tokens) {
            Some((right, tokens)) => Some((
                Expression::Oper(Operator::Multiply, Box::new(expr), Box::new(right)),
                tokens,
            )),
            _ => None,
        }
        Token::Eq => match parse_expression(tokens) {
            Some((right, tokens)) => Some((
                Expression::Oper(Operator::Eq, Box::new(expr), Box::new(right)),
                tokens,
            )),
            _ => None,
        }
        Token::Neq => match parse_expression(tokens) {
            Some((right, tokens)) => Some((
                Expression::Oper(Operator::Neq, Box::new(expr), Box::new(right)),
                tokens,
            )),
            _ => None,
        }
        _ => None,
    }
}

enum Block {
    Statement(Instruction),
    Block(Vec<Instruction>),
}

fn parse_assignment(name: String, tokens: &[Token]) -> Option<(Block, &[Token])> {
    let tokens = if let Some((Token::Assign, tokens)) = tokens.split_first() {
        tokens
    } else {
        return None;
    };
    let (expr, tokens) = match parse_expression(tokens) {
        Some(result) => result,
        None => return None,
    };
    Some((Block::Statement(Instruction::Assign(name, expr)), tokens))
}

fn parse_while(tokens: &[Token], index: usize) -> Option<Block> {
    let (expr, mut tokens) = match parse_expression(tokens) {
        Some(result) => result,
        None => return None,
    };
    let mut block = Vec::new();
    while !tokens.is_empty() {
        match parse_block(tokens, index + block.len()) {
            Some((Block::Statement(state), remaining)) => {
                block.push(state);
                tokens = remaining;
            }
            Some((Block::Block(mut states), remaining)) => {
                block.append(&mut states);
                tokens = remaining;
            }
            None => return None,
        }
    }
    block.push(Instruction::Goto(index));
    let mut output = vec![Instruction::ConditionalJump(
        expr,
        index + 1,
        index + block.len() + 1,
    )];
    output.append(&mut block);
    Some(Block::Block(output))
}

fn parse_block(tokens: &[Token], index: usize) -> Option<(Block, &[Token])> {
    match tokens.split_first() {
        Some((Token::Variable(name), tokens)) => parse_assignment(name.clone(), tokens),
        Some((Token::While(inner), tokens)) => parse_while(inner, index).map(|x| (x, tokens)),
        _ => return None,
    }
}

fn parse_var<'a>(name: String, tokens: &'a [Token], state: &mut State) -> Option<&'a [Token]> {
    let (first, tokens) = match tokens.split_first() {
        Some((first, tokens)) => (first, tokens),
        None => return None,
    };
    if *first != Token::Assign {
        return None;
    };
    let (expr, tokens) = match parse_expression(tokens) {
        Some(result) => result,
        None => return None,
    };
    state.insert(name, expr.evaluate(state, &State::new()));
    Some(tokens)
}

fn parse_function<'a>(name: String, tokens: &'a [Token], state: &mut State) -> Option<&'a [Token]> {
    let mut tokens = match tokens.split_first() {
        Some((Token::StartParen, tokens)) => tokens,
        _ => return None,
    };
    let mut args = vec![];
    loop {
        match tokens.split_first() {
            Some((Token::EndParen, remaining)) => {
                tokens = remaining;
                break;
            }
            Some((Token::Variable(name), remaining)) => {
                args.push(name.clone());
                tokens = remaining;
            }
            _ => return None,
        }
    }
    while let Some((Token::NewLine, remaining)) = tokens.split_first() {
        tokens = remaining;
    }
    let mut statements = vec![];
    while let Some((block, remaining)) = parse_block(tokens, statements.len()) {
        match block {
            Block::Statement(statement) => statements.push(statement),
            Block::Block(mut new_statements) => statements.append(&mut new_statements),
        };
        tokens = remaining;
    }
    let (ret_val, tokens) = match parse_expression(tokens) {
        Some((ret_val, tokens)) => (ret_val, tokens),
        None => return None,
    };
    state.insert(
        name,
        Value::Function(Box::new(Function::from_raw(args, statements, ret_val))),
    );
    Some(tokens)
}

fn parse_value<'a>(tokens: &'a [Token], state: &mut State) -> Option<&'a [Token]> {
    let (first, tokens) = match tokens.split_first() {
        Some((first, tokens)) => (first, tokens),
        None => return None,
    };
    match first {
        Token::Variable(name) => parse_var(name.clone(), tokens, state),
        Token::Function(name) => parse_function(name.clone(), tokens, state),
        Token::NewLine => Some(tokens),
        _ => None,
    }
}

impl State {
    /// Creates a new State containing no data.
    pub fn new() -> Self {
        State {
            variables: HashMap::new(),
        }
    }
    /// Parses given string as a script, and returns the corresponding script
    pub fn from_str(script: &str) -> Self {
        let mut lexer = PreToken::lexer(script);
        Self::from_tokens(&pre_tokens_to_tokens(&mut lexer))
    }
    // This constructs the State from the
    fn from_tokens(mut tokens: &[Token]) -> Self {
        let mut state = State::new();
        while let Some(remaining) = parse_value(tokens, &mut state) {
            tokens = remaining;
        }
        state
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
    fn call_function() {
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
        let mut program = State::from_str(
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
        assert_eq!(program, by_hand);
        assert_eq!(program.run("main"), Some(Value::Integer(21)));
        program.insert(String::from("x"), Value::Integer(8));
        assert_eq!(program.run("main"), Some(Value::Integer(34)));
    }

    #[test]
    fn tokenize() {
        use Token::*;
        let tokens = pre_tokens_to_tokens(&mut PreToken::lexer(
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
        ));
        assert_eq!(
            tokens,
            vec![
                Variable(String::from("x")),
                Assign,
                Literal(Value::Integer(7)),
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
                While(vec![
                    Variable(String::from("x")),
                    Neq,
                    Literal(Value::Integer(0)),
                    NewLine,
                    Variable(String::from("x")),
                    Assign,
                    Variable(String::from("x")),
                    Subtract,
                    Literal(Value::Integer(1)),
                    NewLine,
                    Variable(String::from("temp")),
                    Assign,
                    Variable(String::from("y")),
                    Add,
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
                ]),
                Variable(String::from("y")),
                NewLine
            ]
        );
    }
}
