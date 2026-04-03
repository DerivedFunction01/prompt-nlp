from __future__ import annotations
import html
import json
import random
import shlex
import randomname



class TextFormatter:
    _payload_token = "__PAYLOAD__"


    def _get_id(self) -> str:
    # returns e.g. "sleek-voxel" — use just the noun half for single words
        return randomname.get_name().split("-")[1]


    def _get_name(self, min_parts: int = 1, max_parts: int = 2) -> str:
        if random.randint(min_parts, max_parts) == 1:
            return randomname.get_name().split("-")[1]
        return randomname.get_name().replace("-", "_")


    def _get_class_name(self) -> str:
        """PascalCase class name from 1-2 vocab words."""
        parts = [self._get_id().capitalize() for _ in range(random.randint(1, 2))]
        return "".join(parts)

    def _get_int(self, lo: int = 1, hi: int = 99) -> int:
        return random.randint(lo, hi)

    def _get_timeout(self) -> int:
        return random.choice([50, 100, 150, 200, 250, 500, 1000])

    @staticmethod
    def _indent(lines: list[str], spaces: int = 4) -> str:
        pad = " " * spaces
        return "\n".join(f"{pad}{line}" for line in lines)

    @staticmethod
    def _rand_bool() -> bool:
        return bool(random.getrandbits(1))

    @staticmethod
    def _quoted(text: str) -> str:
        return json.dumps(text, ensure_ascii=False)

    def _materialize(self, template: str, text: str, mode: str = "raw") -> str:
        if mode == "json":
            value = json.dumps(text, ensure_ascii=False)
        elif mode == "xml":
            value = html.escape(text, quote=True)
        elif mode == "shell":
            value = shlex.quote(text)
        elif mode == "sql":
            value = text.replace("'", "''")
        else:
            value = text
        return template.replace(self._payload_token, value)

    def _span_probe(self) -> str:
        """
        Generate a unique probe token that should be safe to embed in code/text.
        """
        return f"__PAYLOAD_{random.getrandbits(32):08x}__"

    def render(self, method: str, text: str) -> str:
        """
        Render a formatter output as plain text.

        This calls the formatter directly, so it is only safe when the formatter
        does not duplicate the payload.
        """
        fn = getattr(self, f"_{method}", None)
        if fn is None:
            raise ValueError(f"Unknown formatter: {method!r}")
        return fn(text)

    def code_format(self, method: str, text: str) -> dict[str, object]:
        """
        Render a formatter output and return the exact payload span.

        The formatter is first rendered with a unique probe token, which is then
        replaced with the real payload. This keeps the payload single-use and
        avoids having to duplicate the original text in the fsormatter template.
        """
        fn = getattr(self, f"_{method}", None)
        if fn is None:
            raise ValueError(f"Unknown formatter: {method!r}")

        probe = self._span_probe()
        templated = fn(probe)
        occurrences = templated.count(probe)
        if occurrences != 1:
            raise ValueError(
                f"Formatter {method!r} must place the payload exactly once; "
                f"found {occurrences} occurrences of the probe token."
            )

        start = templated.index(probe)
        rendered = templated.replace(probe, text, 1)
        span = (start, start + len(text))
        return {"text": rendered, "span": span, "method": method}

    # ------------------------------------------------------------------ #
    #  PYTHON                                                              #
    # ------------------------------------------------------------------ #

    def _python(self, text: str) -> str:
        fname = self._get_name()
        var = self._get_name()
        arg = self._get_name()
        cls = self._get_class_name()
        method = self._get_name()
        q = self._quoted(text)
        lines = text.splitlines() or [text]
        line_repr = json.dumps(lines, ensure_ascii=False)
        logger = self._get_name()

        variants = [
            # simple function + call
            "\n".join(
                [
                    f"def {fname}({arg}):",
                    self._indent([f"{var} = {arg}", f"return {var}"]),
                    "",
                    f"result = {fname}({q})",
                    f"print(result)",
                ]
            ),
            # class with __init__ and method
            "\n".join(
                [
                    f"class {cls}:",
                    self._indent(
                        [
                            f"def __init__(self, {arg}):",
                            f"    self.{var} = {arg}",
                            "",
                            f"def {method}(self):",
                            f"    return self.{var}",
                        ]
                    ),
                    "",
                    f"obj = {cls}({q})",
                    f"print(obj.{method}())",
                ]
            ),
            # list comprehension / join
            "\n".join(
                [
                    f"{var} = {line_repr}",
                    f"output = '\\n'.join(str(item) for item in {var})",
                    f"print(output)",
                ]
            ),
            # logging-style
            "\n".join(
                [
                    "import logging",
                    f"logging.basicConfig(level=logging.INFO)",
                    f"{logger} = logging.getLogger(__name__)",
                    f"{logger}.info({q})",
                ]
            ),
            # try/except wrapper
            "\n".join(
                [
                    f"def {fname}():",
                    self._indent(
                        [
                            "try:",
                            f"    {var} = {q}",
                            f"    print({var})",
                            "except Exception as e:",
                            f"    print(f'Error: {{e}}')",
                        ]
                    ),
                    f"{fname}()",
                ]
            ),
            # conditional + f-string
            "\n".join(
                [
                    f"{var} = {q}",
                    f"if {var}:",
                    self._indent([f"print(f'Value: {{{var}}}')"]),
                    "else:",
                    self._indent(["print('empty')"]),
                ]
            ),
            # dict lookup
            "\n".join(
                [
                    f"data = {{'{self._get_id()}': {q}}}",
                    f"{var} = data.get('{self._get_id()}', '')",
                    f"print({var})",
                ]
            ),
        ]
        return random.choice(variants)

    # ------------------------------------------------------------------ #
    #  JAVASCRIPT / TYPESCRIPT                                            #
    # ------------------------------------------------------------------ #

    def _javascript(self, text: str) -> str:
        fname = self._get_name()
        var = self._get_name()
        arr = self._get_name()
        key = self._get_id()
        q = self._quoted(text)
        timeout = self._get_timeout()

        variants = [
            # arrow function
            f"const {fname} = ({var}) => {{\n  console.log({var});\n}};\n{fname}({q});",
            # async/await stub
            "\n".join(
                [
                    f"async function {fname}() {{",
                    f"  const {var} = await Promise.resolve({q});",
                    f"  console.log({var});",
                    "}",
                    f"{fname}();",
                ]
            ),
            # object destructuring
            "\n".join(
                [
                    f"const {{ {var} }} = {{ {var}: {q} }};",
                    f"console.log({var});",
                ]
            ),
            # array + join
            f"const {arr} = {q}.split(' ');\nconsole.log({arr}.join('-'));",
            # conditional + template literal
            "\n".join(
                [
                    f"const {var} = {q};",
                    f"if ({var}.length > 0) {{",
                    f"  console.log(`Output: ${{{var}}}`);",
                    "}",
                ]
            ),
            # setTimeout
            f"setTimeout(() => {{\n  console.log({q});\n}}, {timeout});",
            # map/filter chain
            "\n".join(
                [
                    f"const {arr} = {q}.split('').filter(Boolean);",
                    f"const {var} = {arr}.map(c => c).join('');",
                    f"console.log({var});",
                ]
            ),
            # object with method
            "\n".join(
                [
                    f"const {var} = {{",
                    f"  {key}: {q},",
                    f"  show() {{ console.log(this.{key}); }}",
                    "};",
                    f"{var}.show();",
                ]
            ),
        ]
        return random.choice(variants)

    def _typescript(self, text: str) -> str:
        fname = self._get_name()
        var = self._get_name()
        cls = self._get_class_name()
        iface = self._get_class_name()
        prop = self._get_id()
        q = self._quoted(text)

        variants = [
            # typed function
            "\n".join(
                [
                    f"function {fname}({var}: string): void {{",
                    f"  console.log({var});",
                    "}",
                    f"{fname}({q});",
                ]
            ),
            # interface + class
            "\n".join(
                [
                    f"interface {iface} {{",
                    f"  {prop}: string;",
                    "}",
                    "",
                    f"const obj: {iface} = {{ {prop}: {q} }};",
                    f"console.log(obj.{prop});",
                ]
            ),
            # generic function
            "\n".join(
                [
                    f"function {fname}<T>({var}: T): T {{",
                    f"  return {var};",
                    "}",
                    f"console.log({fname}({q}));",
                ]
            ),
            # enum-like const
            "\n".join(
                [
                    f"const {var} = {{",
                    f"  VALUE: {q},",
                    "} as const;",
                    f"console.log({var}.VALUE);",
                ]
            ),
        ]
        return random.choice(variants)

    # ------------------------------------------------------------------ #
    #  JAVA                                                                #
    # ------------------------------------------------------------------ #

    def _java(self, text: str) -> str:
        method = self._get_name()
        cls = self._get_class_name()
        var = self._get_name()
        q = self._quoted(text)

        variants = [
            # main class
            "\n".join(
                [
                    f"public class {cls} {{",
                    self._indent(
                        [
                            "public static void main(String[] args) {",
                            f"    String {var} = {q};",
                            f"    System.out.println({var});",
                            "}",
                        ]
                    ),
                    "}",
                ]
            ),
            # method with return
            "\n".join(
                [
                    f"class {cls} {{",
                    self._indent(
                        [
                            f"public String {method}() {{",
                            f"    String {var} = {q};",
                            f"    return {var};",
                            "}",
                        ]
                    ),
                    "}",
                ]
            ),
            # Optional chain
            f"Optional.ofNullable({q}).ifPresent(System.out::println);",
            # Thread lambda
            f"new Thread(() -> System.out.println({q})).start();",
            # StringBuilder
            "\n".join(
                [
                    f"StringBuilder sb = new StringBuilder();",
                    f"sb.append({q});",
                    f"System.out.println(sb.toString());",
                ]
            ),
            # try/catch
            "\n".join(
                [
                    f"try {{",
                    self._indent(
                        [
                            f"String {var} = {q};",
                            f"System.out.println({var});",
                        ]
                    ),
                    "} catch (Exception e) {",
                    self._indent(["e.printStackTrace();"]),
                    "}",
                ]
            ),
        ]
        return random.choice(variants)

    # ------------------------------------------------------------------ #
    #  C#                                                                  #
    # ------------------------------------------------------------------ #

    def _csharp(self, text: str) -> str:
        cls = self._get_class_name()
        method = self._get_name()
        var = self._get_name()
        ns = self._get_class_name()
        q = self._quoted(text)

        variants = [
            # namespace + class + main
            "\n".join(
                [
                    f"namespace {ns} {{",
                    self._indent(
                        [
                            f"class {cls} {{",
                            "    static void Main(string[] args) {",
                            f"        string {var} = {q};",
                            f"        Console.WriteLine({var});",
                            "    }",
                            "}",
                        ]
                    ),
                    "}",
                ]
            ),
            # property + method
            "\n".join(
                [
                    f"public class {cls} {{",
                    self._indent(
                        [
                            f"public string {method.title().replace('_','')}{{ get; set; }} = {q};",
                            "",
                            "public void Display() {",
                            f"    Console.WriteLine({method.title().replace('_','')});",
                            "}",
                        ]
                    ),
                    "}",
                ]
            ),
            # LINQ-style
            "\n".join(
                [
                    f"var {var} = new List<string> {{ {q} }};",
                    f"var result = {var}.Where(s => s.Length > 0).FirstOrDefault();",
                    f"Console.WriteLine(result);",
                ]
            ),
            # string interpolation
            "\n".join(
                [
                    f"string {var} = {q};",
                    f'Console.WriteLine($"Value: {{{var}}}");',
                ]
            ),
        ]
        return random.choice(variants)

    # ------------------------------------------------------------------ #
    #  GO                                                                  #
    # ------------------------------------------------------------------ #

    def _go(self, text: str) -> str:
        fn = self._get_name()
        var = self._get_name()
        pkg = self._get_id()
        q = self._quoted(text)

        variants = [
            # main package
            "\n".join(
                [
                    f"package {pkg}",
                    "",
                    'import "fmt"',
                    "",
                    "func main() {",
                    self._indent([f"{var} := {q}", f"fmt.Println({var})"]),
                    "}",
                ]
            ),
            # named function
            "\n".join(
                [
                    f"package {pkg}",
                    "",
                    'import "fmt"',
                    "",
                    f"func {fn}(s string) {{",
                    self._indent([f"fmt.Println(s)"]),
                    "}",
                    "",
                    "func main() {",
                    self._indent([f"{fn}({q})"]),
                    "}",
                ]
            ),
            # goroutine
            "\n".join(
                [
                    'import "fmt"',
                    "",
                    f"go func() {{",
                    self._indent([f"fmt.Println({q})"]),
                    "}()",
                ]
            ),
        ]
        return random.choice(variants)

    # ------------------------------------------------------------------ #
    #  RUBY                                                                #
    # ------------------------------------------------------------------ #

    def _ruby(self, text: str) -> str:
        method = self._get_name()
        var = self._get_name()
        cls = self._get_class_name()
        q = self._quoted(text)

        variants = [
            # simple puts
            f"{var} = {q}\nputs {var}",
            # method def
            "\n".join(
                [
                    f"def {method}(input)",
                    f"  puts input",
                    "end",
                    "",
                    f"{method}({q})",
                ]
            ),
            # class with initialize
            "\n".join(
                [
                    f"class {cls}",
                    f"  def initialize(val)",
                    f"    @val = val",
                    f"  end",
                    "",
                    f"  def display",
                    f"    puts @val",
                    f"  end",
                    "end",
                    "",
                    f"obj = {cls}.new({q})",
                    f"obj.display",
                ]
            ),
            # block
            "\n".join(
                [
                    f"[{q}].each do |item|",
                    f"  puts item",
                    "end",
                ]
            ),
        ]
        return random.choice(variants)

    # ------------------------------------------------------------------ #
    #  RUST                                                                #
    # ------------------------------------------------------------------ #

    def _rust(self, text: str) -> str:
        fn = self._get_name()
        var = self._get_name()
        struct = self._get_class_name()
        field = self._get_id()
        q = self._quoted(text)

        variants = [
            # main
            "\n".join(
                [
                    "fn main() {",
                    self._indent([f"let {var} = {q};", f'println!("{{}}", {var});']),
                    "}",
                ]
            ),
            # function + call
            "\n".join(
                [
                    f"fn {fn}(s: &str) {{",
                    self._indent([f'println!("{{}}", s);']),
                    "}",
                    "",
                    "fn main() {",
                    self._indent([f"{fn}({q});"]),
                    "}",
                ]
            ),
            # struct + impl
            "\n".join(
                [
                    f"struct {struct} {{",
                    self._indent([f"{field}: String,"]),
                    "}",
                    "",
                    f"impl {struct} {{",
                    self._indent(
                        [
                            f"fn display(&self) {{",
                            f'    println!("{{}}", self.{field});',
                            "}",
                        ]
                    ),
                    "}",
                    "",
                    "fn main() {",
                    self._indent(
                        [
                            f"let obj = {struct} {{ {field}: {q}.to_string() }};",
                            "obj.display();",
                        ]
                    ),
                    "}",
                ]
            ),
        ]
        return random.choice(variants)

    # ------------------------------------------------------------------ #
    #  PHP                                                                 #
    # ------------------------------------------------------------------ #

    def _php(self, text: str) -> str:
        fn = self._get_name()
        var = self._get_name()
        cls = self._get_class_name()
        prop = self._get_id()
        q = self._quoted(text)

        variants = [
            # simple echo
            f"<?php\n${var} = {q};\necho ${var};\n?>",
            # function
            "\n".join(
                [
                    "<?php",
                    f"function {fn}($input) {{",
                    f"    echo $input;",
                    "}",
                    f"{fn}({q});",
                    "?>",
                ]
            ),
            # class
            "\n".join(
                [
                    "<?php",
                    f"class {cls} {{",
                    self._indent(
                        [
                            f"private string ${prop};",
                            "",
                            f"public function __construct(string ${prop}) {{",
                            f"    $this->{prop} = ${prop};",
                            "}",
                            "",
                            "public function display(): void {",
                            f"    echo $this->{prop};",
                            "}",
                        ]
                    ),
                    "}",
                    f"$obj = new {cls}({q});",
                    "$obj->display();",
                    "?>",
                ]
            ),
        ]
        return random.choice(variants)

    # ------------------------------------------------------------------ #
    #  SQL                                                                 #
    # ------------------------------------------------------------------ #

    def _sql(self, text: str) -> str:
        table = self._get_name()
        col1, col2 = self._get_id(), self._get_id()
        alias = self._get_id()
        rid = self._get_int(1, 500)
        q_text = text.replace("'", "''")  # basic SQL escaping

        variants = [
            f"SELECT '{q_text}' AS {alias};",
            f"SELECT {col1}, {col2} FROM {table} WHERE {col1} = '{q_text}' LIMIT 1;",
            "\n".join(
                [
                    f"INSERT INTO {table} ({col1}, {col2})",
                    f"VALUES ('{q_text}', {rid});",
                ]
            ),
            "\n".join(
                [
                    f"UPDATE {table}",
                    f"SET {col1} = '{q_text}'",
                    f"WHERE id = {rid};",
                ]
            ),
            "\n".join(
                [
                    f"SELECT t.{col1}",
                    f"FROM {table} t",
                    f"WHERE t.{col2} = '{q_text}'",
                    f"ORDER BY t.{col1} ASC;",
                ]
            ),
        ]
        return random.choice(variants)

    # ------------------------------------------------------------------ #
    #  XML                                                                 #
    # ------------------------------------------------------------------ #

    def _xml(self, text: str) -> str:
        root = self._get_name()
        child = self._get_name()
        attr = self._get_id()
        rid = self._get_int()

        variants = [
            f"<{root} id='{rid}'>\n  <{child}>{self._payload_token}</{child}>\n</{root}>",
            f"<{root}><{child} {attr}='{rid}'>{self._payload_token}</{child}></{root}>",
            "\n".join(
                [
                    '<?xml version="1.0" encoding="UTF-8"?>',
                    f"<{root}>",
                    self._indent([f"<{child}>{self._payload_token}</{child}>"]),
                    f"</{root}>",
                ]
            ),
            "\n".join(
                [
                    f"<{root}>",
                    self._indent(
                        [
                            f'<{child} {attr}="{rid}">{self._payload_token}</{child}>',
                        ]
                    ),
                    f"</{root}>",
                ]
            ),
        ]
        return self._materialize(random.choice(variants), text, mode="xml")

    # ------------------------------------------------------------------ #
    #  MARKDOWN                                                           #
    # ------------------------------------------------------------------ #

    def _markdown(self, text: str) -> str:
        units = [part for part in text.splitlines() if part.strip()]
        if "\n\n" in text or len(units) >= 3:
            units = [part.strip() for part in text.split("\n\n") if part.strip()]
        if not units:
            units = [text]

        block_variants = [
            f"**{text}**",
            f"*{text}*",
            f"> {text}",
            f"`{text}`",
            f"### {text}",
        ]
        line_styles = [
            lambda line: f"- {line}",
            lambda line: f"**{line}**",
            lambda line: f"*{line}*",
            lambda line: f"> {line}",
        ]

        if len(units) > 1:
            formatted = [random.choice(line_styles)(line) for line in units]
            return "\n".join(formatted)
        return random.choice(block_variants)

    # ------------------------------------------------------------------ #
    #  BASH / SHELL / SCRIPT                                              #
    # ------------------------------------------------------------------ #

    def _bash(self, text: str) -> str:
        var = self._get_name()
        fn = self._get_name()
        q = self._quoted(text)
        lines = text.splitlines() or [text]

        variants = [
            # variable + echo
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    f"{var}={q}",
                    f'echo "${{{var}}}"',
                ]
            ),
            # printf
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    f"printf '%s\\n' {q}",
                ]
            ),
            # heredoc function
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    f"{fn}() {{",
                    "  while IFS= read -r line; do",
                    '    echo "$line"',
                    "  done <<'EOF'",
                    *lines,
                    "EOF",
                    "}",
                    f"{fn}",
                ]
            ),
            # conditional
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    f"{var}={q}",
                    f'if [ -n "${{{var}}}" ]; then',
                    f'  echo "${{{var}}}"',
                    "fi",
                ]
            ),
            # set -e + trap
            "\n".join(
                [
                    "#!/usr/bin/env bash",
                    "set -euo pipefail",
                    f"trap 'echo error' ERR",
                    f"echo {q}",
                ]
            ),
        ]
        return random.choice(variants)

    def _script(self, text: str) -> str:
        """Alias kept for backward compatibility — delegates to bash."""
        return self._bash(text)

    # ------------------------------------------------------------------ #
    #  JSON                                                                #
    # ------------------------------------------------------------------ #

    def _json(self, text: str) -> str:
        key1 = self._get_id()
        key2 = self._get_id()
        key3 = self._get_id()
        rid = self._get_int()

        variants = [
            # flat object
            json.dumps({key1: text}, ensure_ascii=False, indent=2),
            # nested object
            json.dumps({key1: {key2: text, key3: rid}}, ensure_ascii=False, indent=2),
            # array of objects
            json.dumps(
                [{key1: text}, {key1: text[::-1]}], ensure_ascii=False, indent=2
            ),
            # metadata-style
            json.dumps(
                {
                    "id": rid,
                    key1: text,
                    "meta": {key2: True, key3: rid * 2},
                },
                ensure_ascii=False,
                indent=2,
            ),
        ]
        return random.choice(variants)
