# scripting-language
A scripting language intented to be used with rustmania, still in its early stages

# Grammar

Here is a grammar for the currently implemented features:

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

Statement := [Assignment | While]

While := [while][Expression][Statement]*[end while]```
