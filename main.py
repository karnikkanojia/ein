from ein.parsing import ParsedExpression, validate_pair

input_tree = ParsedExpression("b (h 2)")
output_tree = ParsedExpression("b h 2")
print(validate_pair(input_tree, output_tree))  # should not raise

