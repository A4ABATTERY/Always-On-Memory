"""
Unit tests for the Lexical Symbol Index parser.
Covers Python (AST), JS/TS (regex), Go, Rust, and edge cases.
"""

import unittest
from lexical_parser import extract_symbols


PYTHON_SOURCE = '''\
import os

CONSTANT_A = 42
not_a_constant = "ignored"

class MyClass:
    def method(self):
        pass

def top_level_func(x, y):
    def nested():
        pass
    return x + y

async def async_handler(request):
    pass
'''

JS_SOURCE = """\
export async function handleRequest(req) {}
class ApiRouter {}
const MY_CONST = 42;
const arrowFn = (x) => x + 1;
"""

GO_SOURCE = """\
package main

func (r *Router) HandleHTTP(w ResponseWriter, req *Request) {}

func TopLevelFunc() {}

type MyStruct struct {}

const MY_CONST = 10
"""

RUST_SOURCE = """\
pub fn public_fn(x: i32) -> i32 { x }
fn private_fn() {}
pub struct MyStruct {}
pub enum MyEnum { A, B }
pub const MAX_SIZE: usize = 100;
"""

SYNTAX_ERROR_PYTHON = "def broken(\n    # unclosed"

EMPTY_SOURCE = ""
DATA_FILE = "key=value\nother=123\n"


class TestLexicalParserPython(unittest.TestCase):

    def setUp(self):
        self.symbols = extract_symbols(PYTHON_SOURCE, ".py")
        self.names = {s["name"] for s in self.symbols}

    def test_extracts_class(self):
        self.assertIn("MyClass", self.names)

    def test_extracts_top_level_function(self):
        self.assertIn("top_level_func", self.names)

    def test_extracts_async_function(self):
        self.assertIn("async_handler", self.names)

    def test_extracts_upper_case_constant(self):
        self.assertIn("CONSTANT_A", self.names)

    def test_excludes_lowercase_assignment(self):
        self.assertNotIn("not_a_constant", self.names)

    def test_nested_function_included(self):
        # ast.walk finds nested functions too — nested() is inside top_level_func
        self.assertIn("nested", self.names)

    def test_symbol_types_correct(self):
        types = {s["name"]: s["type"] for s in self.symbols}
        self.assertEqual(types["MyClass"], "class")
        self.assertEqual(types["top_level_func"], "function")
        self.assertEqual(types["CONSTANT_A"], "constant")

    def test_line_numbers_present(self):
        for s in self.symbols:
            self.assertIsNotNone(s["line_no"], f"Missing line_no for {s['name']}")
            self.assertGreater(s["line_no"], 0)

    def test_syntax_error_returns_empty(self):
        result = extract_symbols(SYNTAX_ERROR_PYTHON, ".py")
        self.assertEqual(result, [])

    def test_empty_source_returns_empty(self):
        result = extract_symbols(EMPTY_SOURCE, ".py")
        self.assertEqual(result, [])


class TestLexicalParserJsTs(unittest.TestCase):

    def setUp(self):
        self.symbols = extract_symbols(JS_SOURCE, ".ts")
        self.names = {s["name"] for s in self.symbols}

    def test_extracts_async_function(self):
        self.assertIn("handleRequest", self.names)

    def test_extracts_class(self):
        self.assertIn("ApiRouter", self.names)

    def test_extracts_upper_constant(self):
        self.assertIn("MY_CONST", self.names)

    def test_extracts_arrow_function(self):
        self.assertIn("arrowFn", self.names)

    def test_jsx_extension_supported(self):
        result = extract_symbols("function MyComponent() {}", ".jsx")
        names = {s["name"] for s in result}
        self.assertIn("MyComponent", names)


class TestLexicalParserGo(unittest.TestCase):

    def setUp(self):
        self.symbols = extract_symbols(GO_SOURCE, ".go")
        self.names = {s["name"] for s in self.symbols}

    def test_extracts_function(self):
        self.assertIn("TopLevelFunc", self.names)

    def test_extracts_method_receiver_function(self):
        self.assertIn("HandleHTTP", self.names)

    def test_extracts_struct(self):
        self.assertIn("MyStruct", self.names)

    def test_extracts_constant(self):
        self.assertIn("MY_CONST", self.names)


class TestLexicalParserRust(unittest.TestCase):

    def setUp(self):
        self.symbols = extract_symbols(RUST_SOURCE, ".rs")
        self.names = {s["name"] for s in self.symbols}

    def test_extracts_pub_fn(self):
        self.assertIn("public_fn", self.names)

    def test_extracts_private_fn(self):
        self.assertIn("private_fn", self.names)

    def test_extracts_struct(self):
        self.assertIn("MyStruct", self.names)

    def test_extracts_enum(self):
        self.assertIn("MyEnum", self.names)

    def test_extracts_constant(self):
        self.assertIn("MAX_SIZE", self.names)


class TestLexicalParserEdgeCases(unittest.TestCase):

    def test_unsupported_extension_returns_empty(self):
        result = extract_symbols(DATA_FILE, ".env")
        self.assertEqual(result, [])

    def test_unknown_extension_returns_empty(self):
        result = extract_symbols("something = 1", ".xyz")
        self.assertEqual(result, [])

    def test_no_symbols_python_returns_empty(self):
        result = extract_symbols("x = 1\ny = 2\n", ".py")
        # No UPPER_CASE, no functions, no classes
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
