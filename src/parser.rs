use crate::ast::{Expression, Function, Instruction, Operator, State, Value};
use logos::{Lexer, Logos};

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
    #[token = "\""]
    Quotation,
    #[regex = "[a-zA-Z][a-zA-Z0-9]*"]
    Identifier,
    #[end]
    End,
    #[error]
    Error,
}

fn parse_operator(
    lexer: &mut Lexer<Token, &str>,
    left_precedence: u8,
    expr: Expression,
) -> Result<Expression, String> {
    let op_token = lexer.token;
    let (op, right_precedence) = match op_token {
        Token::Add => (Operator::Add, 2),
        Token::Subtract => (Operator::Subtract, 2),
        Token::Mul => (Operator::Multiply, 3),
        Token::Eq => (Operator::Eq, 1),
        Token::Neq => (Operator::Neq, 1),
        _ => return Ok(expr),
    };
    if left_precedence > right_precedence {
        Ok(expr)
    } else {
        lexer.advance();
        Ok({
            let expr = Expression::Oper(
                op,
                Box::new(expr),
                Box::new(parse_expression(lexer, right_precedence)?),
            );
            parse_operator(lexer, left_precedence, expr)?
        })
    }
}

fn parse_expression(
    lexer: &mut Lexer<Token, &str>,
    left_precedence: u8,
) -> Result<Expression, String> {
    let variable = lexer.token;
    let mut expr = match variable {
        Token::Integer => {
            Expression::Lit(Value::Integer(lexer.slice().parse().expect(lexer.slice())))
        }
        Token::Float => Expression::Lit(Value::Float(lexer.slice().parse().unwrap())),
        Token::Identifier => Expression::Var(lexer.slice().to_string()),
        Token::Quotation => {
            lexer.advance();
            let start = lexer.range().start;
            while lexer.token != Token::Quotation {
                lexer.advance();
            }
            let end = lexer.range().start;
            Expression::Lit(Value::Text(lexer.source[start..end].to_string()))
        }
        _ => return Err(lexer.slice().to_string()),
    };
    lexer.advance();
    if lexer.token == Token::StartParen {
        lexer.advance();
        let mut args = vec![];
        while lexer.token != Token::EndParen {
            match parse_expression(lexer, 0) {
                Ok(expression) => args.push(expression),
                Err(_) => break,
            }
        }
        lexer.advance();
        expr = Expression::Call(Box::new(expr), args);
    }
    parse_operator(lexer, left_precedence, expr)
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
    let expr = parse_expression(lexer, 0)?;
    Ok(Block::Statement(Instruction::Assign(name, expr)))
}

fn parse_if(lexer: &mut Lexer<Token, &str>, index: usize) -> Result<Block, String> {
    lexer.advance();
    let expr = parse_expression(lexer, 0)?;
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
    let expr = parse_expression(lexer, 0)?;
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
    let expr = parse_expression(lexer, 0)?;
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
    let ret_val = parse_expression(&mut last_lexer, 0)?;
    *lexer = last_lexer;
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

pub(crate) fn from_str(script: &str) -> Result<State, String> {
    let mut state = State::new();
    let mut lexer = Token::lexer(script);
    while lexer.token != Token::End {
        parse_value(&mut lexer, &mut state)?;
    }
    Ok(state)
}
