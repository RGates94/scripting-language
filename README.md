# Scripting Language
This project is an early stage scripting language intended to be used with RustMania.  The primary goal is making the language easy to learn, read, and integrate with Rust.

# Grammar

This is a grammar for the currently implemented features:

```
Program := [Declaration]*

Declaration := [Assignment | Function]

Assignment := [Identifier][=][Expression][\n]

Identifier := [a-zA-Z][a-zA-Z0-9]*

Expression := [Identifier | Literal | [Expression][Operator][Expression] ]

Literal := [Integer | Float]

Integer := [0-9]+

Float := [0-9]+[.][0-9]+

Operator := [+ | - | *]

Function := [fn][Identifier][(][Identifier]*[)][\n][Statement]*[Expression]

Statement := [Assignment | While | If]

While := [while][Expression][Statement]*[end]

If := [if][Expression][Statement}*[else]?[Statement]*[end]```
