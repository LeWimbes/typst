use std::iter::Peekable;
use std::str::Chars;
use unicode_xid::UnicodeXID;

use crate::size::Size;
use super::span::{Position, Span, Spanned};

use self::Token::*;
use self::TokenizationMode::*;


/// A minimal semantic entity of source code.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Token<'s> {
    /// One or more whitespace characters. The contained `usize` denotes the
    /// number of newlines that were contained in the whitespace.
    Space(usize),

    /// A line comment with inner string contents `//<str>\n`.
    LineComment(&'s str),
    /// A block comment with inner string contents `/*<str>*/`. The comment
    /// can contain nested block comments.
    BlockComment(&'s str),

    /// A function invocation.
    Function {
        /// The header string:
        /// ```typst
        /// [header: args][body]
        ///  ^^^^^^^^^^^^
        /// ```
        header: &'s str,
        /// The spanned body string:
        /// ```typst
        /// [header][hello *world*]
        ///          ^^^^^^^^^^^^^
        /// ```
        ///
        /// The span includes the brackets while the string does not.
        body: Option<Spanned<&'s str>>,
        /// Whether the last closing bracket was present.
        /// - `[func]` or `[func][body]` => terminated
        /// - `[func` or `[func][body` => not terminated
        terminated: bool,
    },

    /// A left parenthesis in a function header: `(`.
    LeftParen,
    /// A right parenthesis in a function header: `)`.
    RightParen,
    /// A left brace in a function header: `{`.
    LeftBrace,
    /// A right brace in a function header: `}`.
    RightBrace,

    /// A colon in a function header: `:`.
    Colon,
    /// A comma in a function header: `:`.
    Comma,
    /// An equals sign in a function header: `=`.
    Equals,

    /// An identifier in a function header: `center`.
    ExprIdent(&'s str),
    /// A quoted string in a function header: `"..."`.
    ExprStr {
        /// The string inside the quotes.
        string: &'s str,
        /// Whether the closing quote was present.
        terminated: bool
    },
    /// A number in a function header: `3.14`.
    ExprNumber(f64),
    /// A size in a function header: `12pt`.
    ExprSize(Size),
    /// A boolean in a function header: `true | false`.
    ExprBool(bool),

    /// A star in body-text.
    Star,
    /// An underscore in body-text.
    Underscore,
    /// A backtick in body-text.
    Backtick,

    /// Any other consecutive string.
    Text(&'s str),

    /// Things that are not valid in the context they appeared in.
    Invalid(&'s str),
}

impl<'s> Token<'s> {
    /// The natural-language name for this token for use in error messages.
    pub fn name(self) -> &'static str {
        match self {
            Space(_)        => "space",
            LineComment(_)  => "line comment",
            BlockComment(_) => "block comment",
            Function { .. } => "function",
            LeftParen       => "opening paren",
            RightParen      => "closing paren",
            LeftBrace       => "opening brace",
            RightBrace      => "closing brace",
            Colon           => "colon",
            Comma           => "comma",
            Equals          => "equals sign",
            ExprIdent(_)    => "identifier",
            ExprStr { .. }  => "string",
            ExprNumber(_)   => "number",
            ExprSize(_)     => "size",
            ExprBool(_)     => "boolean",
            Star            => "star",
            Underscore      => "underscore",
            Backtick        => "backtick",
            Text(_)         => "invalid identifier",
            Invalid("]")    => "closing bracket",
            Invalid("*/")   => "end of block comment",
            Invalid(_)      => "invalid token",
        }
    }
}

/// An iterator over the tokens of a string of source code.
#[derive(Debug)]
pub struct Tokens<'s> {
    src: &'s str,
    mode: TokenizationMode,
    iter: Peekable<Chars<'s>>,
    position: Position,
    index: usize,
}

/// Whether to tokenize in header mode which yields expression, comma and
/// similar tokens or in body mode which yields text and star, underscore,
/// backtick tokens.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
#[allow(missing_docs)]
pub enum TokenizationMode {
    Header,
    Body,
}

impl<'s> Tokens<'s> {
    /// Create a new token iterator with the given mode where the first token
    /// span starts an the given `start` position.
    pub fn new(start: Position, src: &'s str, mode: TokenizationMode) -> Tokens<'s> {
        Tokens {
            src,
            mode,
            iter: src.chars().peekable(),
            position: start,
            index: 0,
        }
    }

    /// The index in the string at which the last token ends and next token will
    /// start.
    pub fn index(&self) -> usize {
        self.index
    }

    /// The line-colunn position in the source at which the last token ends and
    /// next token will start. This position is
    pub fn pos(&self) -> Position {
        self.position
    }
}

impl<'s> Iterator for Tokens<'s> {
    type Item = Spanned<Token<'s>>;

    /// Parse the next token in the source code.
    fn next(&mut self) -> Option<Spanned<Token<'s>>> {
        let start = self.pos();
        let first = self.eat()?;

        let token = match first {
            // Comments.
            '/' if self.peek() == Some('/') => self.parse_line_comment(),
            '/' if self.peek() == Some('*') => self.parse_block_comment(),
            '*' if self.peek() == Some('/') => { self.eat(); Invalid("*/") }

            // Whitespace.
            c if c.is_whitespace() => self.parse_whitespace(start),

            // Functions.
            '[' => self.parse_function(start),
            ']' => Invalid("]"),

            // Syntactic elements in function headers.
            '(' if self.mode == Header => LeftParen,
            ')' if self.mode == Header => RightParen,
            '{' if self.mode == Header => LeftBrace,
            '}' if self.mode == Header => RightBrace,
            ':' if self.mode == Header => Colon,
            ',' if self.mode == Header => Comma,
            '=' if self.mode == Header => Equals,

            // String values.
            '"' if self.mode == Header => self.parse_string(),

            // Style toggles.
            '*' if self.mode == Body => Star,
            '_' if self.mode == Body => Underscore,
            '`' if self.mode == Body => Backtick,

            // An escaped thing.
            '\\' => self.parse_escaped(),

            // Expressions or just strings.
            c => {
                let text = self.read_string_until(|n| {
                    match n {
                        c if c.is_whitespace() => true,
                        '\\' | '[' | ']' | '*' | '_' | '`' | ':' | '=' |
                        ',' | '"' | '/' => true,
                        _ => false,
                    }
                }, false, -(c.len_utf8() as isize), 0).0;

                if self.mode == Header {
                    self.parse_expr(text)
                } else {
                    Text(text)
                }
            }
        };

        let end = self.pos();
        let span = Span { start, end };

        Some(Spanned { v: token, span })
    }
}

impl<'s> Tokens<'s> {
    fn parse_line_comment(&mut self) -> Token<'s> {
        LineComment(self.read_string_until(is_newline_char, false, 1, 0).0)
    }

    fn parse_block_comment(&mut self) -> Token<'s> {
        enum Last { Slash, Star, Other }

        self.eat();

        let mut depth = 0;
        let mut last = Last::Other;

        // Find the first `*/` that does not correspond to a nested `/*`.
        // Remove the last two bytes to obtain the raw inner text without `*/`.
        BlockComment(self.read_string_until(|n| {
            match n {
                '/' => match last {
                    Last::Star if depth == 0 => return true,
                    Last::Star => depth -= 1,
                    _ => last = Last::Slash
                }
                '*' => match last {
                    Last::Slash => depth += 1,
                    _ => last = Last::Star,
                }
                _ => last = Last::Other,
            }

            false
        }, true, 0, -2).0)
    }

    fn parse_whitespace(&mut self, start: Position) -> Token<'s> {
        self.read_string_until(|n| !n.is_whitespace(), false, 0, 0);
        let end = self.pos();

        Space(end.line - start.line)
    }

    fn parse_function(&mut self, start: Position) -> Token<'s> {
        let (header, terminated) = self.read_function_part();
        self.eat();

        if self.peek() != Some('[') {
            return Function { header, body: None, terminated };
        }

        let body_start = self.pos() - start;
        self.eat();

        let (body, terminated) = self.read_function_part();
        self.eat();

        let body_end = self.pos();
        let span = Span::new(body_start, body_end);

        Function { header, body: Some(Spanned { v: body, span }), terminated }
    }

    fn read_function_part(&mut self) -> (&'s str, bool) {
        let mut escaped = false;
        let mut in_string = false;
        let mut depth = 0;

        self.read_string_until(|n| {
            match n {
                '"' if !escaped => in_string = !in_string,
                '[' if !escaped && !in_string => depth += 1,
                ']' if !escaped && !in_string => {
                    if depth == 0 {
                        return true;
                    } else {
                        depth -= 1;
                    }
                }
                '\\' => escaped = !escaped,
                _ => escaped = false,
            }

            false
        }, false, 0, 0)
    }

    fn parse_string(&mut self) -> Token<'s> {
        let mut escaped = false;
        let (string, terminated) = self.read_string_until(|n| {
            match n {
                '"' if !escaped => return true,
                '\\' => escaped = !escaped,
                _ => escaped = false,
            }

            false
        }, true, 0, -1);
        ExprStr { string, terminated }
    }

    fn parse_escaped(&mut self) -> Token<'s> {
        fn is_escapable(c: char) -> bool {
            match c {
                '\\' | '[' | ']' | '*' | '_' | '`' | '/' => true,
                _ => false,
            }
        }

        let c = self.peek().unwrap_or('n');
        let string = if is_escapable(c) {
            let index = self.index();
            self.eat();
            &self.src[index .. index + c.len_utf8()]
        } else {
            "\\"
        };

        match self.mode {
            Header => Invalid(string),
            Body => Text(string),
        }
    }

    fn parse_expr(&mut self, text: &'s str) -> Token<'s> {
        if let Ok(b) = text.parse::<bool>() {
            ExprBool(b)
        } else if let Ok(num) = text.parse::<f64>() {
            ExprNumber(num)
        } else if let Some(num) = parse_percentage(text) {
            ExprNumber(num / 100.0)
        } else if let Ok(size) = text.parse::<Size>() {
            ExprSize(size)
        } else if is_identifier(text) {
            ExprIdent(text)
        } else {
            Invalid(text)
        }
    }

    fn read_string_until<F>(
        &mut self,
        mut f: F,
        eat_match: bool,
        offset_start: isize,
        offset_end: isize,
    ) -> (&'s str, bool) where F: FnMut(char) -> bool {
        let start = ((self.index() as isize) + offset_start) as usize;
        let mut matched = false;

        while let Some(c) = self.peek() {
            if f(c) {
                matched = true;
                if eat_match {
                    self.eat();
                }
                break;
            }

            self.eat();
        }

        let mut end = self.index();
        if matched {
            end = ((end as isize) + offset_end) as usize;
        }

        (&self.src[start .. end], matched)
    }

    fn eat(&mut self) -> Option<char> {
        let c = self.iter.next()?;
        self.index += c.len_utf8();

        if is_newline_char(c) && !(c == '\r' && self.peek() == Some('\n')) {
            self.position.line += 1;
            self.position.column = 0;
        } else {
            self.position.column += 1;
        }

        Some(c)
    }

    fn peek(&mut self) -> Option<char> {
        self.iter.peek().copied()
    }
}

fn parse_percentage(text: &str) -> Option<f64> {
    if text.ends_with('%') {
        text[.. text.len() - 1].parse::<f64>().ok()
    } else {
        None
    }
}

/// Whether this character denotes a newline.
pub fn is_newline_char(character: char) -> bool {
    match character {
        // Line Feed, Vertical Tab, Form Feed, Carriage Return.
        '\x0A' ..= '\x0D' => true,
        // Next Line, Line Separator, Paragraph Separator.
        '\u{0085}' | '\u{2028}' | '\u{2029}' => true,
        _ => false,
    }
}

/// Whether this word is a valid identifier.
pub fn is_identifier(string: &str) -> bool {
    let mut chars = string.chars();

    match chars.next() {
        Some('-') => {}
        Some(c) if UnicodeXID::is_xid_start(c) => {}
        _ => return false,
    }

    while let Some(c) = chars.next() {
        match c {
            '.' | '-' => {}
            c if UnicodeXID::is_xid_continue(c) => {}
            _ => return false,
        }
    }

    true
}


#[cfg(test)]
mod tests {
    use super::*;

    use Token::{
        Space as S,
        LineComment as LC, BlockComment as BC,
        LeftParen as LP, RightParen as RP,
        LeftBrace as LB, RightBrace as RB,
        ExprIdent as Id, ExprNumber as Num, ExprBool as Bool,
        Text as T,
    };

    #[allow(non_snake_case)]
    fn Str(string: &'static str, terminated: bool) -> Token<'static> {
        Token::ExprStr { string, terminated }
    }

    /// Test whether the given string tokenizes into the given list of tokens.
    macro_rules! t {
        ($m:expr, $s:expr => [$(($sl:tt:$sc:tt, $el:tt:$ec:tt, $t:expr)),* $(,)?]) => {
            let tokens = Tokens::new(Position::ZERO, $s, $m).collect::<Vec<_>>();
            assert_eq!(tokens, vec![$(Spanned {
                span: Span::new(Position::new($sl, $sc), Position::new($el, $ec)),
                v: $t
            }),*]);

        };

        ($m:expr, $s:expr => [$($t:expr),* $(,)?]) => {
            let tokens = Tokens::new(Position::ZERO, $s, $m)
                .map(Spanned::value)
                .collect::<Vec<_>>();
            assert_eq!(tokens, vec![$($t),*]);
        };
    }

    /// Parse a function token.
    macro_rules! func {
        ($header:expr, Some(($sl:tt:$sc:tt, $el:tt:$ec:tt, $body:expr)), $terminated:expr) => {
            Function {
                header: $header,
                body: Some(Spanned {
                    span: Span::new(Position::new($sl, $sc), Position::new($el, $ec)),
                    v: $body,
                }),
                terminated: $terminated,
            }
        };
        ($header:expr, None, $terminated:expr) => {
            Function { header: $header, body: None, terminated: $terminated }
        }
    }

    #[test]
    fn tokenize_whitespace() {
        t!(Body, ""             => []);
        t!(Body, " "            => [S(0)]);
        t!(Body, "    "         => [S(0)]);
        t!(Body, "\t"           => [S(0)]);
        t!(Body, "  \t"         => [S(0)]);
        t!(Body, "\n"           => [S(1)]);
        t!(Body, "\n "          => [S(1)]);
        t!(Body, "  \n"         => [S(1)]);
        t!(Body, "  \n   "      => [S(1)]);
        t!(Body, "\r\n"         => [S(1)]);
        t!(Body, "  \n\t \n  "  => [S(2)]);
        t!(Body, "\n\r"         => [S(2)]);
        t!(Body, " \r\r\n \x0D" => [S(3)]);
    }

    #[test]
    fn tokenize_comments() {
        t!(Body, "a // bc\n "        => [T("a"), S(0), LC(" bc"),  S(1)]);
        t!(Body, "a //a//b\n "       => [T("a"), S(0), LC("a//b"), S(1)]);
        t!(Body, "a //a//b\r\n"      => [T("a"), S(0), LC("a//b"), S(1)]);
        t!(Body, "a //a//b\n\nhello" => [T("a"), S(0), LC("a//b"), S(2), T("hello")]);
        t!(Body, "/**/"              => [BC("")]);
        t!(Body, "_/*_/*a*/*/"       => [Underscore, BC("_/*a*/")]);
        t!(Body, "/*/*/"             => [BC("/*/")]);
        t!(Body, "abc*/"             => [T("abc"), Invalid("*/")]);
    }

    #[test]
    fn tokenize_header_only_tokens() {
        t!(Body, "\"hi\""              => [T("\"hi"), T("\"")]);
        t!(Body, "a: b"                => [T("a"), T(":"), S(0), T("b")]);
        t!(Body, "c=d, "               => [T("c"), T("=d"), T(","), S(0)]);
        t!(Header, "["                 => [func!("", None, false)]);
        t!(Header, "]"                 => [Invalid("]")]);
        t!(Header, "(){}:=,"           => [LP, RP, LB, RB, Colon, Equals, Comma]);
        t!(Header, "a:b"               => [Id("a"), Colon, Id("b")]);
        t!(Header, "="                 => [Equals]);
        t!(Header, ","                 => [Comma]);
        t!(Header, r#""hello\"world""# => [Str(r#"hello\"world"#, true)]);
        t!(Header, r#""hi", 12pt"#     => [Str("hi", true), Comma, S(0), ExprSize(Size::pt(12.0))]);
        t!(Header, "a: true, x=1"      => [Id("a"), Colon, S(0), Bool(true), Comma, S(0), Id("x"), Equals, Num(1.0)]);
        t!(Header, "120%"              => [Num(1.2)]);
        t!(Header, "🌓, 🌍,"           => [Invalid("🌓"), Comma, S(0), Invalid("🌍"), Comma]);
    }

    #[test]
    fn tokenize_body_only_tokens() {
        t!(Body, "_*`"           => [Underscore, Star, Backtick]);
        t!(Body, "[func]*bold*"  => [func!("func", None, true), Star, T("bold"), Star]);
        t!(Body, "hi_you_ there" => [T("hi"), Underscore, T("you"), Underscore, S(0), T("there")]);
        t!(Header, "_*`"         => [Invalid("_"), Invalid("*"), Invalid("`")]);
    }

    #[test]
    fn tokenize_nested_functions() {
        t!(Body, "[f: [=][*]]"    => [func!("f: [=][*]", None, true)]);
        t!(Body, "[_][[,],],"     => [func!("_", Some((0:3, 0:9, "[,],")), true), T(",")]);
        t!(Body, "[=][=][=]"      => [func!("=", Some((0:3, 0:6, "=")), true), func!("=", None, true)]);
        t!(Body, "[=][[=][=][=]]" => [func!("=", Some((0:3, 0:14, "[=][=][=]")), true)]);
    }

    #[test]
    fn tokenize_escaped_symbols() {
        t!(Body, r"\\" => [T(r"\")]);
        t!(Body, r"\[" => [T("[")]);
        t!(Body, r"\]" => [T("]")]);
        t!(Body, r"\*" => [T("*")]);
        t!(Body, r"\_" => [T("_")]);
        t!(Body, r"\`" => [T("`")]);
        t!(Body, r"\/" => [T("/")]);
    }

    #[test]
    fn tokenize_unescapable_symbols() {
        t!(Body, r"\a"     => [T("\\"), T("a")]);
        t!(Body, r"\:"     => [T(r"\"), T(":")]);
        t!(Body, r"\="     => [T(r"\"), T("=")]);
        t!(Header, r"\\\\" => [Invalid("\\"), Invalid("\\")]);
        t!(Header, r"\a"   => [Invalid("\\"), Id("a")]);
        t!(Header, r"\:"   => [Invalid(r"\"), Colon]);
        t!(Header, r"\="   => [Invalid(r"\"), Equals]);
        t!(Header, r"\,"   => [Invalid(r"\"), Comma]);
    }

    #[test]
    fn tokenize_with_spans() {
        t!(Body, "hello"          => [(0:0, 0:5, T("hello"))]);
        t!(Body, "ab\r\nc"        => [(0:0, 0:2, T("ab")), (0:2, 1:0, S(1)), (1:0, 1:1, T("c"))]);
        t!(Body, "[x = \"(1)\"]*" => [(0:0, 0:11, func!("x = \"(1)\"", None, true)), (0:11, 0:12, Star)]);
        t!(Body, "// ab\r\n\nf"   => [(0:0, 0:5, LC(" ab")), (0:5, 2:0, S(2)), (2:0, 2:1, T("f"))]);
        t!(Body, "/*b*/_"         => [(0:0, 0:5, BC("b")), (0:5, 0:6, Underscore)]);
        t!(Header, "a=10"         => [(0:0, 0:1, Id("a")), (0:1, 0:2, Equals), (0:2, 0:4, Num(10.0))]);
    }
}
