"""
DeepSeek API Server - Tool Parser
"""

import re
import json
from config import VALID_TOOLS

def _extract_xml_tags(text: str) -> list:
    """Parse tool call XML from text. Supports:
    1. <tool>name</tool><json>{...}</json>
    2. <function_call name="X"><args>JSON</args></function_call> (legacy)
    3. <tool><name>X</name><parameter name="Y" string="true">value</parameter>...</tool>
    4. <tool_name>...</tool_name> (bare tool tags from CodeAI)
    5. <tool><tool_call name="X"><parameter name="Y">value</parameter>...</tool_call></tool>
    6. <tool><tool_call>X</tool_call><parameter name="Y">value</parameter>...</tool>
    7. <tool>{JSON}</tool> (raw JSON with name field)
    8. <tool><tool_name name="X">X</tool_name><json>{...}</json></tool>
    9. <invoke name="X"><parameter>...</parameter></invoke>
    10. <tool><invoke name="X"><parameter>...</parameter></invoke></tool> (trip)
    11. <tool><tool_name>X</tool_name><child>v</child>...</tool> (direct child params)
    12. <tool><tool_name>X</tool_name><parameter name="Y">v</parameter></tool>
    13. <tool tool_name="X"><json>{...}</json></tool> (tool_name as attribute)
    14. <tool><tool_name>X</tool_name><input>{JSON}</input></tool>
    15. <tool><tool_name>X</tool_name><arguments><x>v</x></arguments></tool>
    16. <tool name="X" ... /> (attribute-only self-closing)
    17. <tool tool_name="X" json='{...}' /> (self-closing attrs)
    18. <invoke name="X" ... /> (invoke self-closing)
    19. <tools><tool>...</tool>...</tools> (multi-tool wrapper)
    """
    # Strip markdown code blocks: XML trong ``` ... ``` là ví dụ, không phải tool call thật
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    # Strip inline code chứa tool name ảo (ví dụ từ code parser):
    # `<tool>name</tool>`, `<name>X</name>`, `<NAME>...</NAME>`, v.v.
    for fake in ('name', 'NAME', 'X', 'N', 'tool_name', 'TÊN', 'TEN', 'tên_tool'):
        text = re.sub(rf'`<[^>]*{fake}[^>]*>`', '', text)
    tools = []
    
    # Format 1: <tool>name</tool><json>{...}</json>
    tool_pattern = re.compile(
        r'<tool>\s*(\w+)\s*</tool>\s*<json>(.*?)</json>',
        re.DOTALL
    )
    for match in tool_pattern.finditer(text):
        tool_name = match.group(1)
        args_str = match.group(2).strip()
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            args = {}
        tools.append({"name": tool_name, "arguments": args})
    if tools:
        return tools

    # Format 1b: <tool>\n<json>{...}</json> (no tool name, possibly no closing </tool>)
    # Infer tool name from JSON keys
    no_name_pattern = re.compile(
        r'<tool>\s*<json>(.*?)</json>',
        re.DOTALL
    )
    KEY_TO_TOOL = {
        "questions": "AskUserQuestion",
        "command": "bash",
        "file_path": "read",
        "plan": "UpdatePlan",
        "query": "WebSearch",
        "old_string": "edit",
    }
    for match in no_name_pattern.finditer(text):
        args_str = match.group(1).strip()
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            args = {}
        tool_name = "unknown"
        for key, tname in KEY_TO_TOOL.items():
            if key in args:
                tool_name = tname
                break
        tools.append({"name": tool_name, "arguments": args})
    if tools:
        return tools



    # Format 9: <tool><parameter name="KEY">value</parameter>...</tool> (no name, no name attr)
    # Infer tool name from parameter keys (command→bash, file_path→read)
    tool_params_only_pattern = re.compile(
        r'<tool>\s*(<parameter\s+name\s*=\s*"\w+"[^>]*>\s*.*?\s*</parameter>\s*)+</tool>',
        re.DOTALL
    )
    for match in tool_params_only_pattern.finditer(text):
        params_block = match.group(0)
        args = {}
        param_pattern = re.compile(
            r'<parameter\s+name\s*=\s*"(\w+)"[^>]*>\s*(.*?)\s*</parameter>',
            re.DOTALL
        )
        for pm in param_pattern.finditer(params_block):
            pname = pm.group(1)
            pvalue = pm.group(2).strip()
            type_match = re.search(r'string\s*=\s*"(true|false)"', pm.group(0))
            if type_match and type_match.group(1) == "false":
                try:
                    pvalue = json.loads(pvalue)
                except (json.JSONDecodeError, ValueError):
                    pass
            args[pname] = pvalue
        tool_name = "unknown"
        for key, tname in KEY_TO_TOOL.items():
            if key in args:
                tool_name = tname
                break
        tools.append({"name": tool_name, "arguments": args})
    if tools:
        return tools

    # Format 10: <tool><tool_call>NAME</tool_call><json>{...}</json></tool>
    tool_call_tag_pattern = re.compile(
        r'<tool>\s*<tool_call>(\w+)</tool_call>\s*<json>(.*?)</json>\s*</tool>',
        re.DOTALL
    )
    for match in tool_call_tag_pattern.finditer(text):
        tool_name = match.group(1)
        args_str = match.group(2).strip()
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            args = {}
        tools.append({"name": tool_name, "arguments": args})
    if tools:
        return tools

    # Format 10b: <tool><tool_call name="NAME"><parameter name="KEY" string="true/false">value</parameter>...</tool_call></tool>
    tool_call_attr_param_pattern = re.compile(
        r'<tool>\s*<tool_call\s+name\s*=\s*"(\w+)"\s*>(.*?)</tool_call>\s*</tool>',
        re.DOTALL
    )
    for match in tool_call_attr_param_pattern.finditer(text):
        tool_name = match.group(1)
        params_block = match.group(2)
        args = {}
        param_pattern = re.compile(
            r'<parameter\s+name\s*=\s*"(\w+)"[^>]*>\s*(.*?)\s*</parameter>',
            re.DOTALL
        )
        for pm in param_pattern.finditer(params_block):
            pname = pm.group(1)
            pvalue = pm.group(2).strip()
            type_match = re.search(r'string\s*=\s*"(true|false)"', pm.group(0))
            if type_match and type_match.group(1) == "false":
                try:
                    pvalue = json.loads(pvalue)
                except (json.JSONDecodeError, ValueError):
                    pass
            args[pname] = pvalue
        tools.append({"name": tool_name, "arguments": args})
    if tools:
        return tools

    # Format 10c: <tool><tool_call>NAME</tool_call><parameter name="KEY">value</parameter>...</tool>
    tool_call_param_pattern = re.compile(
        r'<tool>\s*<tool_call>(\w+)</tool_call>\s*((?:\s*<parameter\s+name\s*=\s*"\w+"[^>]*>\s*.*?\s*</parameter>\s*)+)</tool>',
        re.DOTALL
    )
    for match in tool_call_param_pattern.finditer(text):
        tool_name = match.group(1)
        params_block = match.group(2)
        args = {}
        param_pattern2 = re.compile(
            r'<parameter\s+name\s*=\s*"(\w+)"[^>]*>\s*(.*?)\s*</parameter>',
            re.DOTALL
        )
        for pm in param_pattern2.finditer(params_block):
            pname = pm.group(1)
            pvalue = pm.group(2).strip()
            type_match = re.search(r'string\s*=\s*"(true|false)"', pm.group(0))
            if type_match and type_match.group(1) == "false":
                try:
                    pvalue = json.loads(pvalue)
                except (json.JSONDecodeError, ValueError):
                    pass
            args[pname] = pvalue
        tools.append({"name": tool_name, "arguments": args})
    if tools:
        return tools

    # Format 11: <tool><tool>NAME</tool><parameter>key</parameter><parameter>val</parameter>...</tool>
    # Sequential key-value parameter pairs
    tool_nested_tag_pattern = re.compile(
        r'<tool>\s*<tool>(\w+)</tool>\s*((?:\s*<parameter>[^<]*</parameter>\s*)+)</tool>',
        re.DOTALL
    )
    for match in tool_nested_tag_pattern.finditer(text):
        tool_name = match.group(1)
        params_block = match.group(2)
        param_values = re.findall(r'<parameter>\s*(.*?)\s*</parameter>', params_block, re.DOTALL)
        args = {}
        for i in range(0, len(param_values) - 1, 2):
            key = param_values[i].strip()
            val = param_values[i + 1].strip()
            try:
                val = json.loads(val)
            except (json.JSONDecodeError, ValueError):
                pass
            args[key] = val
        tools.append({"name": tool_name, "arguments": args})
    if tools:
        return tools

    # Format 11b: <tool><tool>NAME</tool><parameter>...</parameter> (missing </tool>)
    tool_nested_no_close_pattern = re.compile(
        r'<tool>\s*<tool>(\w+)</tool>\s*((?:\s*<parameter>[^<]*</parameter>\s*)+)',
        re.DOTALL
    )
    for match in tool_nested_no_close_pattern.finditer(text):
        tool_name = match.group(1)
        params_block = match.group(2)
        param_values = re.findall(r'<parameter>\s*(.*?)\s*</parameter>', params_block, re.DOTALL)
        args = {}
        for i in range(0, len(param_values) - 1, 2):
            key = param_values[i].strip()
            val = param_values[i + 1].strip()
            try:
                val = json.loads(val)
            except (json.JSONDecodeError, ValueError):
                pass
            args[key] = val
        tools.append({"name": tool_name, "arguments": args})
    if tools:
        return tools

    # Format 3: <tool><name>X</name><parameter ...>value</parameter>...</tool>
    tool_block_pattern = re.compile(
        r'<tool>\s*<name>(\w+)</name>(.*?)</tool>',
        re.DOTALL
    )
    for match in tool_block_pattern.finditer(text):
        tool_name = match.group(1)
        params_block = match.group(2)
        args = {}
        param_pattern = re.compile(
            r'<parameter\s+name\s*=\s*"(\w+)"[^>]*>\s*(.*?)\s*</parameter>',
            re.DOTALL
        )
        for pm in param_pattern.finditer(params_block):
            pname = pm.group(1)
            pvalue = pm.group(2).strip()
            # Parse string="true" boolean or string="false" for non-string values
            type_match = re.search(r'string\s*=\s*"(true|false)"', pm.group(0))
            if type_match and type_match.group(1) == "false":
                try:
                    pvalue = json.loads(pvalue)
                except (json.JSONDecodeError, ValueError):
                    pass
            args[pname] = pvalue
        tools.append({"name": tool_name, "arguments": args})
    if tools:
        return tools

    # Format 6: <tool><tool_name><json>{...}</json></tool_name></tool>
    # CodeAI nested XML with json wrapper: <tool><read><json>{"file_path":"..."}</json></read></tool>
    IGN = re.DOTALL | re.IGNORECASE  # hỗ trợ <Bash>, <Read>, v.v.
    tool_names = sorted(VALID_TOOLS)  # sync với VALID_TOOLS
    for tname in tool_names:
        json_inner_pattern = re.compile(
            rf'<tool>\s*<{tname}>\s*<json>(.*?)</json>\s*</{tname}>\s*</tool>',
            IGN
        )
        for match in json_inner_pattern.finditer(text):
            args_str = match.group(1).strip()
            try:
                args = json.loads(args_str)
            except (json.JSONDecodeError, ValueError):
                args = {}
            tools.append({"name": tname.lower(), "arguments": args})
    if tools:
        return tools

    # Format 7: <tool><tool_name><json>{...}</json></tool> (missing </tool_name>)
    # Variant of Format 6: <tool><read><json>{\"file_path\":\"...\"}</json></tool>
    for tname in tool_names:
        no_close_name_pattern = re.compile(
            rf'<tool>\s*<{tname}>\s*<json>(.*?)</json>\s*</tool>',
            IGN
        )
        for match in no_close_name_pattern.finditer(text):
            args_str = match.group(1).strip()
            try:
                args = json.loads(args_str)
            except (json.JSONDecodeError, ValueError):
                args = {}
            tools.append({"name": tname.lower(), "arguments": args})
    if tools:
        return tools

    # Format 5: <tool><tool_name><param>value</param>...</tool_name></tool>
    # CodeAI nested XML: <tool><bash><command>...</command><description>...</description></bash></tool>
    for tname in tool_names:
        inner_tool_pattern = re.compile(
            rf'<tool>\s*<{tname}>(.*?)</{tname}>\s*</tool>',
            IGN
        )
        for match in inner_tool_pattern.finditer(text):
            inner_content = match.group(1)
            args = {}
            child_pattern = re.compile(r'<(\w+)>(.*?)</\1>', IGN)
            for cm in child_pattern.finditer(inner_content):
                key = cm.group(1).lower()
                value = cm.group(2).strip()
                try:
                    value = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    pass
                args[key] = value
            tools.append({"name": tname.lower(), "arguments": args})
    if tools:
        return tools

    # Format 5b: <tool><tool_name><param>value</param>...</tool> (no closing tool_name tag)
    # Variant where </tool> closes both wrapper and inner element.
    # Example: <tool><bash><command>ls</command><description>list</description></tool>
    for tname in tool_names:
        no_close_inner_pattern = re.compile(
            rf'<tool>\s*<{tname}>(.*?)</tool>',
            IGN
        )
        for match in no_close_inner_pattern.finditer(text):
            inner_content = match.group(1)
            args = {}
            child_pattern = re.compile(r'<(\w+)>(.*?)</\1>', IGN)
            for cm in child_pattern.finditer(inner_content):
                key = cm.group(1).lower()
                value = cm.group(2).strip()
                try:
                    value = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    pass
                args[key] = value
            tools.append({"name": tname.lower(), "arguments": args})
    if tools:
        return tools

    # Format 4: Bare tool tags like <bash>...</bash>, <read>...</read>, <write>...</write>
    # These are CodeAI-style: <tool_name>content</tool_name> - no parameters
    for tname in tool_names:
        bare_pattern = re.compile(
            rf'<{tname}>\s*(.*?)\s*</{tname}>',
            IGN
        )
        for match in bare_pattern.finditer(text):
            content = match.group(1).strip()
            # Try parse as JSON, otherwise treat as raw text
            try:
                args = json.loads(content)
            except json.JSONDecodeError:
                args = {"content": content} if content else {}
            tools.append({"name": tname.lower(), "arguments": args})
    if tools:
        return tools

    # Format 8: <tool name="TOOL"><parameter name="KEY">value</parameter>...</tool>
    tool_attr_pattern = re.compile(
        r'<tool\s+name\s*=\s*"(\w+)"\s*>(.*?)</tool>',
        re.DOTALL
    )
    for match in tool_attr_pattern.finditer(text):
        tool_name = match.group(1)
        params_block = match.group(2)
        args = {}
        param_pattern = re.compile(
            r'<parameter\s+name\s*=\s*"(\w+)"[^>]*>\s*(.*?)\s*</parameter>',
            re.DOTALL
        )
        for pm in param_pattern.finditer(params_block):
            pname = pm.group(1)
            pvalue = pm.group(2).strip()
            type_match = re.search(r'string\s*=\s*"(true|false)"', pm.group(0))
            if type_match and type_match.group(1) == "false":
                try:
                    pvalue = json.loads(pvalue)
                except (json.JSONDecodeError, ValueError):
                    pass
            args[pname] = pvalue
        tools.append({"name": tool_name, "arguments": args})
    if tools:
        return tools

    # Format 2 (legacy): <function_call name="X"><args>JSON</args></function_call>
    fc_pattern = re.compile(
        r'<function_call\s+name\s*=\s*"(\w+)"\s*>\s*'
        r'<args>(.*?)</args>\s*'
        r'</function_call>',
        re.DOTALL
    )
    for match in fc_pattern.finditer(text):
        tool_name = match.group(1)
        args_str = match.group(2).strip()
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            args = {}
        tools.append({"name": tool_name, "arguments": args})
    if tools:
        return tools

    # Format 2b: <function_call name="X"><args>JSON</args> (missing </function_call>)
    # Fallback when DeepSeek forgets the closing function_call tag
    fc_no_close_pattern = re.compile(
        r'<function_call\s+name\s*=\s*"(\w+)"\s*>\s*<args>(.*?)</args>',
        re.DOTALL
    )
    for match in fc_no_close_pattern.finditer(text):
        tool_name = match.group(1)
        args_str = match.group(2).strip()
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            args = {}
        tools.append({"name": tool_name, "arguments": args})

    # Format 12: <tool>{JSON}</tool> (raw JSON with name/direct tool call fields)
    # Supports both <tool>{"name":"bash","command":"ls",...}</tool>
    # and     <tool>{"name":"bash","description":"...","command":"...","sideEffects":[...]}</tool>
    tool_raw_json_pattern = re.compile(
        r'<tool>\s*(.+?)\s*</tool>',
        re.DOTALL
    )
    for match in tool_raw_json_pattern.finditer(text):
        content = match.group(1).strip()
        # Only handle JSON objects/arrays (skip XML children)
        if not (content.startswith('{') or content.startswith('[')):
            continue
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict):
            # Extract 'name' field for tool name; rest are arguments
            tool_name = data.pop('name', None)
            if tool_name:
                tools.append({"name": tool_name, "arguments": data})
            else:
                # No name field → infer from JSON keys
                for key, tname in KEY_TO_TOOL.items():
                    if key in data:
                        tools.append({"name": tname, "arguments": data})
                        break
        elif isinstance(data, list):
            # Array of tool objects, each with a 'name' field
            for item in data:
                if isinstance(item, dict) and 'name' in item:
                    name = item.pop('name')
                    tools.append({"name": name, "arguments": item})
    if tools:
        return tools

    # Format 13: <tool><tool_name name="X">X</tool_name><json>{...}</json></tool>
    # CodeAI / deepapi multi-tool XML: tool_name tag with name attribute + json body
    #   <tool>
    #     <tool_name name="bash">bash</tool_name>
    #     <json>{"command":"...","description":"...","sideEffects":[...]}</json>
    #   </tool>
    tool_name_tag_json_pattern = re.compile(
        r'<tool>\s*<tool_name\s+name\s*=\s*"(\w+)"[^>]*>\s*\1\s*</tool_name>\s*<json>(.*?)</json>\s*</tool>',
        re.DOTALL
    )
    for match in tool_name_tag_json_pattern.finditer(text):
        tool_name = match.group(1)
        args_str = match.group(2).strip()
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            args = {}
        tools.append({"name": tool_name, "arguments": args})
    if tools:
        return tools

    # Format 13b: <tool><tool_name name="X">X</tool_name><json>{...}</json> (missing </tool>)
    # Loose closing: </tool> may be omitted
    tool_name_tag_json_no_close_pattern = re.compile(
        r'<tool>\s*<tool_name\s+name\s*=\s*"(\w+)"[^>]*>\s*\1\s*</tool_name>\s*<json>(.*?)</json>',
        re.DOTALL
    )
    for match in tool_name_tag_json_no_close_pattern.finditer(text):
        tool_name = match.group(1)
        args_str = match.group(2).strip()
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            args = {}
        tools.append({"name": tool_name, "arguments": args})
    if tools:
        return tools

    # Format 14: <invoke name="NAME"><parameter name="KEY" string="true/false">value</parameter>...</invoke>
    invoke_pattern = re.compile(
        r'<invoke\s+name\s*=\s*"(\w+)"\s*>(.*?)</invoke>',
        re.DOTALL
    )
    for match in invoke_pattern.finditer(text):
        tool_name = match.group(1)
        params_block = match.group(2)
        args = {}
        param_pattern = re.compile(
            r'<parameter\s+name\s*=\s*"(\w+)"[^>]*>\s*(.*?)\s*</parameter>',
            re.DOTALL
        )
        for pm in param_pattern.finditer(params_block):
            pname = pm.group(1)
            pvalue = pm.group(2).strip()
            type_match = re.search(r'string\s*=\s*"(true|false)"', pm.group(0))
            if type_match and type_match.group(1) == "false":
                try:
                    pvalue = json.loads(pvalue)
                except (json.JSONDecodeError, ValueError):
                    pass
            args[pname] = pvalue
        tools.append({"name": tool_name, "arguments": args})
    if tools:
        return tools

    # Format 14b: <invoke name="NAME"><parameter name="KEY">value</parameter>... (missing </invoke>)
    invoke_no_close_pattern = re.compile(
        r'<invoke\s+name\s*=\s*"(\w+)"\s*>'
        r'((?:\s*<parameter\s+name\s*=\s*"\w+"[^>]*>\s*.*?\s*</parameter>\s*)+)',
        re.DOTALL
    )
    for match in invoke_no_close_pattern.finditer(text):
        tool_name = match.group(1)
        params_block = match.group(2)
        args = {}
        param_pattern2 = re.compile(
            r'<parameter\s+name\s*=\s*"(\w+)"[^>]*>\s*(.*?)\s*</parameter>',
            re.DOTALL
        )
        for pm in param_pattern2.finditer(params_block):
            pname = pm.group(1)
            pvalue = pm.group(2).strip()
            type_match = re.search(r'string\s*=\s*"(true|false)"', pm.group(0))
            if type_match and type_match.group(1) == "false":
                try:
                    pvalue = json.loads(pvalue)
                except (json.JSONDecodeError, ValueError):
                    pass
            args[pname] = pvalue
        tools.append({"name": tool_name, "arguments": args})
    if tools:
        return tools

    # Format 15: <tool><tool_name>NAME</tool_name><json>{...}</json></tool>
    # (simple tool_name without name attribute, different from Format 13)
    tool_name_json_pattern = re.compile(
        r'<tool>\s*<tool_name>(\w+)</tool_name>\s*<json>(.*?)</json>\s*</tool>',
        re.DOTALL
    )
    for match in tool_name_json_pattern.finditer(text):
        tool_name = match.group(1)
        args_str = match.group(2).strip()
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            args = {}
        tools.append({"name": tool_name, "arguments": args})
    if tools:
        return tools

    # Format 15b: <tool><tool_name>NAME</tool_name><json>{...}</json> (missing </tool>)
    tool_name_json_no_close_pattern = re.compile(
        r'<tool>\s*<tool_name>(\w+)</tool_name>\s*<json>(.*?)</json>',
        re.DOTALL
    )
    for match in tool_name_json_no_close_pattern.finditer(text):
        tool_name = match.group(1)
        args_str = match.group(2).strip()
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            args = {}
        tools.append({"name": tool_name, "arguments": args})
    if tools:
        return tools

    # Format 16: <tool><invoke name="NAME"><parameter name="KEY">value</parameter>...</invoke></tool>
    # (trip: tool wrapper + invoke + parameters)
    tool_invoke_pattern = re.compile(
        r'<tool>\s*<invoke\s+name\s*=\s*"(\w+)"\s*>(.*?)</invoke>\s*</tool>',
        re.DOTALL
    )
    for match in tool_invoke_pattern.finditer(text):
        tool_name = match.group(1)
        params_block = match.group(2)
        args = {}
        param_pattern3 = re.compile(
            r'<parameter\s+name\s*=\s*"(\w+)"[^>]*>\s*(.*?)\s*</parameter>',
            re.DOTALL
        )
        for pm in param_pattern3.finditer(params_block):
            pname = pm.group(1)
            pvalue = pm.group(2).strip()
            type_match = re.search(r'string\s*=\s*"(true|false)"', pm.group(0))
            if type_match and type_match.group(1) == "false":
                try:
                    pvalue = json.loads(pvalue)
                except (json.JSONDecodeError, ValueError):
                    pass
            args[pname] = pvalue
        tools.append({"name": tool_name, "arguments": args})
    if tools:
        return tools

    # Format 16b: <tool><invoke name="NAME"><parameter>...</parameter></invoke> (missing </tool>)
    tool_invoke_no_close_pattern = re.compile(
        r'<tool>\s*<invoke\s+name\s*=\s*"(\w+)"\s*>(.*?)</invoke>',
        re.DOTALL
    )
    for match in tool_invoke_no_close_pattern.finditer(text):
        tool_name = match.group(1)
        params_block = match.group(2)
        args = {}
        param_pattern4 = re.compile(
            r'<parameter\s+name\s*=\s*"(\w+)"[^>]*>\s*(.*?)\s*</parameter>',
            re.DOTALL
        )
        for pm in param_pattern4.finditer(params_block):
            pname = pm.group(1)
            pvalue = pm.group(2).strip()
            type_match = re.search(r'string\s*=\s*"(true|false)"', pm.group(0))
            if type_match and type_match.group(1) == "false":
                try:
                    pvalue = json.loads(pvalue)
                except (json.JSONDecodeError, ValueError):
                    pass
            args[pname] = pvalue
        tools.append({"name": tool_name, "arguments": args})
    if tools:
        return tools

    # Format 26: <tools><tool>...</tool><tool>...</tool></tools>
    # (multi-tool wrapper, Var 30) — MUST come first (container)
    tools_wrapper = re.compile(
        r'<tools>(.*?)</tools>',
        re.DOTALL
    )
    for match in tools_wrapper.finditer(text):
        inner = match.group(1)
        inner_tools = _extract_xml_tags(inner)
        tools.extend(inner_tools)
    if tools:
        return tools

    # Format 23: <tool tool_name="X" json='{...}' />
    # (self-closing with tool_name + json attrs, Var 14) — before Format 22
    tool_attr_json_selfclose = re.compile(
        r"""<tool\s+tool_name\s*=\s*(?:"(\w+)"|'(\w+)')\s+"""
        r"""json\s*=\s*(?:"([^"]*)"|'([^']*)')\s*/>""",
        re.DOTALL
    )
    for match in tool_attr_json_selfclose.finditer(text):
        tool_name = match.group(1) or match.group(2)
        json_str = match.group(3) or match.group(4)
        try:
            args = json.loads(json_str)
        except json.JSONDecodeError:
            args = {}
        tools.append({"name": tool_name, "arguments": args})
    if tools:
        return tools

    # Format 24: <tool name="X" params='{...}' />
    # (self-closing with name + params attrs, Var 7) — before Format 22
    tool_name_params_selfclose = re.compile(
        r"""<tool\s+name\s*=\s*(?:"(\w+)"|'(\w+)')\s+"""
        r"""params\s*=\s*(?:"([^"]*)"|'([^']*)')\s*/>""",
        re.DOTALL
    )
    for match in tool_name_params_selfclose.finditer(text):
        tool_name = match.group(1) or match.group(2)
        params_str = match.group(3) or match.group(4)
        try:
            args = json.loads(params_str)
        except json.JSONDecodeError:
            args = {}
        tools.append({"name": tool_name, "arguments": args})
    if tools:
        return tools

    # Format 22: <tool name="X" attr1="v1" attr2="v2" />
    # (attribute-only self-closing, Var 3)
    tool_attr_selfclose = re.compile(
        r'<tool\s+([^>]+?)\s*/>',
        re.DOTALL
    )
    for match in tool_attr_selfclose.finditer(text):
        attrs_str = match.group(1)
        attr_pat = re.compile(r"""(\w+)\s*=\s*(?:"([^"]*)"|'([^']*)')""")
        attrs = {}
        for am in attr_pat.finditer(attrs_str):
            akey = am.group(1)
            aval = am.group(2) if am.group(2) is not None else am.group(3)
            attrs[akey] = aval
        if 'name' in attrs:
            tool_name = attrs.pop('name')
            tools.append({"name": tool_name, "arguments": attrs})
        else:
            for key, tname in KEY_TO_TOOL.items():
                if key in attrs:
                    tools.append({"name": tname, "arguments": attrs})
                    break
    if tools:
        return tools

    # Format 25: <invoke name="X" attr1="v1" attr2="v2" />
    # (invoke self-closing with inline attrs, Var 12)
    invoke_attr_selfclose = re.compile(
        r'<invoke\s+([^>]+?)\s*/>',
        re.DOTALL
    )
    for match in invoke_attr_selfclose.finditer(text):
        attrs_str = match.group(1)
        attr_pat = re.compile(r"""(\w+)\s*=\s*(?:"([^"]*)"|'([^']*)')""")
        attrs = {}
        for am in attr_pat.finditer(attrs_str):
            akey = am.group(1)
            aval = am.group(2) if am.group(2) is not None else am.group(3)
            attrs[akey] = aval
        if 'name' in attrs:
            tool_name = attrs.pop('name')
            tools.append({"name": tool_name, "arguments": attrs})
    if tools:
        return tools

    # Format 19: <tool tool_name="X"><json>{...}</json></tool>
    # (tool_name as attribute + json child, Var 17)
    tool_attr_json_pattern = re.compile(
        r'<tool\s+tool_name\s*=\s*"(\w+)"\s*>\s*<json>(.*?)</json>\s*</tool>',
        re.DOTALL
    )
    for match in tool_attr_json_pattern.finditer(text):
        tool_name = match.group(1)
        args_str = match.group(2).strip()
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            args = {}
        tools.append({"name": tool_name, "arguments": args})
    if tools:
        return tools

    # Format 19b: <tool tool_name="X"><json>{...}</json> (missing </tool>)
    tool_attr_json_no_close = re.compile(
        r'<tool\s+tool_name\s*=\s*"(\w+)"\s*>\s*<json>(.*?)</json>',
        re.DOTALL
    )
    for match in tool_attr_json_no_close.finditer(text):
        tool_name = match.group(1)
        args_str = match.group(2).strip()
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            args = {}
        tools.append({"name": tool_name, "arguments": args})
    if tools:
        return tools

    # Format 20: <tool><tool_name>X</tool_name><input>{JSON}</input></tool>
    # (<input> instead of <json>, Var 22) — before Format 17
    tool_input_pattern = re.compile(
        r'<tool>\s*<tool_name>(\w+)</tool_name>\s*<input>(.*?)</input>\s*</tool>',
        re.DOTALL
    )
    for match in tool_input_pattern.finditer(text):
        tool_name = match.group(1)
        args_str = match.group(2).strip()
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            args = {}
        tools.append({"name": tool_name, "arguments": args})
    if tools:
        return tools

    # Format 20b: <tool><tool_name>X</tool_name><input>{JSON}</input> (missing </tool>)
    tool_input_no_close = re.compile(
        r'<tool>\s*<tool_name>(\w+)</tool_name>\s*<input>(.*?)</input>',
        re.DOTALL
    )
    for match in tool_input_no_close.finditer(text):
        tool_name = match.group(1)
        args_str = match.group(2).strip()
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            args = {}
        tools.append({"name": tool_name, "arguments": args})
    if tools:
        return tools

    # Format 21: <tool><tool_name>X</tool_name><arguments><child>v</child></arguments></tool>
    # (<arguments> wrapper, Var 5) — before Format 17
    tool_args_pattern = re.compile(
        r'<tool>\s*<tool_name>(\w+)</tool_name>\s*<arguments>(.*?)</arguments>\s*</tool>',
        re.DOTALL
    )
    for match in tool_args_pattern.finditer(text):
        tool_name = match.group(1)
        inner = match.group(2)
        args = {}
        child_pat = re.compile(r'<(\w+)>\s*(.*?)\s*</\1>', re.DOTALL)
        for cm in child_pat.finditer(inner):
            key = cm.group(1).lower()
            val = cm.group(2).strip()
            try:
                val = json.loads(val)
            except (json.JSONDecodeError, ValueError):
                pass
            args[key] = val
        tools.append({"name": tool_name, "arguments": args})
    if tools:
        return tools

    # Format 21b: <tool><tool_name>X</tool_name><arguments>...</arguments> (missing </tool>)
    tool_args_no_close = re.compile(
        r'<tool>\s*<tool_name>(\w+)</tool_name>\s*<arguments>(.*?)</arguments>',
        re.DOTALL
    )
    for match in tool_args_no_close.finditer(text):
        tool_name = match.group(1)
        inner = match.group(2)
        if re.search(r'</tool>', text[match.end():match.end()+10]):
            continue
        args = {}
        child_pat = re.compile(r'<(\w+)>\s*(.*?)\s*</\1>', re.DOTALL)
        for cm in child_pat.finditer(inner):
            key = cm.group(1).lower()
            val = cm.group(2).strip()
            try:
                val = json.loads(val)
            except (json.JSONDecodeError, ValueError):
                pass
            args[key] = val
        tools.append({"name": tool_name, "arguments": args})
    if tools:
        return tools

    # Format 18: <tool><tool_name>X</tool_name><parameter name="Y">v</parameter>...</tool>
    # (tool_name + parameter tags, Var 10) — before Format 17
    tool_name_param_pattern = re.compile(
        r'<tool>\s*<tool_name>(\w+)</tool_name>\s*'
        r'((?:\s*<parameter\s+name\s*=\s*"\w+"[^>]*>\s*.*?\s*</parameter>\s*)+)\s*</tool>',
        re.DOTALL
    )
    for match in tool_name_param_pattern.finditer(text):
        tool_name = match.group(1)
        params_block = match.group(2)
        args = {}
        pp = re.compile(r'<parameter\s+name\s*=\s*"(\w+)"[^>]*>\s*(.*?)\s*</parameter>', re.DOTALL)
        for pm in pp.finditer(params_block):
            pname = pm.group(1)
            pvalue = pm.group(2).strip()
            tm = re.search(r'string\s*=\s*"(true|false)"', pm.group(0))
            if tm and tm.group(1) == "false":
                try:
                    pvalue = json.loads(pvalue)
                except (json.JSONDecodeError, ValueError):
                    pass
            args[pname] = pvalue
        tools.append({"name": tool_name, "arguments": args})
    if tools:
        return tools

    # Format 18b: <tool><tool_name>X</tool_name><parameter...>... (missing </tool>)
    tool_name_param_no_close = re.compile(
        r'<tool>\s*<tool_name>(\w+)</tool_name>\s*'
        r'((?:\s*<parameter\s+name\s*=\s*"\w+"[^>]*>\s*.*?\s*</parameter>\s*)+)',
        re.DOTALL
    )
    for match in tool_name_param_no_close.finditer(text):
        tool_name = match.group(1)
        params_block = match.group(2)
        if re.search(r'</tool>', text[match.end():match.end()+10]):
            continue
        args = {}
        pp = re.compile(r'<parameter\s+name\s*=\s*"(\w+)"[^>]*>\s*(.*?)\s*</parameter>', re.DOTALL)
        for pm in pp.finditer(params_block):
            pname = pm.group(1)
            pvalue = pm.group(2).strip()
            tm = re.search(r'string\s*=\s*"(true|false)"', pm.group(0))
            if tm and tm.group(1) == "false":
                try:
                    pvalue = json.loads(pvalue)
                except (json.JSONDecodeError, ValueError):
                    pass
            args[pname] = pvalue
        tools.append({"name": tool_name, "arguments": args})
    if tools:
        return tools

    # Format 17: <tool><tool_name>X</tool_name><child1>v1</child1>...</tool>
    # (tool_name tag + direct child params, Var 2/23) — LAST among tool_name formats
    tool_name_child_pattern = re.compile(
        r'<tool>\s*<tool_name>(\w+)</tool_name>(.*?)</tool>',
        re.DOTALL
    )
    for match in tool_name_child_pattern.finditer(text):
        tool_name = match.group(1)
        inner = match.group(2)
        args = {}
        child_pat = re.compile(r'<(\w+)>\s*(.*?)\s*</\1>', re.DOTALL)
        for cm in child_pat.finditer(inner):
            key = cm.group(1).lower()
            val = cm.group(2).strip()
            try:
                val = json.loads(val)
            except (json.JSONDecodeError, ValueError):
                pass
            args[key] = val
        tools.append({"name": tool_name, "arguments": args})
    if tools:
        return tools

    # Format 17b: <tool><tool_name>X</tool_name><child>v</child>... (missing </tool>)
    tool_name_child_no_close = re.compile(
        r'<tool>\s*<tool_name>(\w+)</tool_name>((?:\s*<\w+>[^<]*</\w+>\s*)+)',
        re.DOTALL
    )
    for match in tool_name_child_no_close.finditer(text):
        tool_name = match.group(1)
        inner = match.group(2)
        if re.search(r'</tool>', text[match.end():match.end()+10]):
            continue
        args = {}
        child_pat = re.compile(r'<(\w+)>\s*(.*?)\s*</\1>', re.DOTALL)
        for cm in child_pat.finditer(inner):
            key = cm.group(1).lower()
            val = cm.group(2).strip()
            try:
                val = json.loads(val)
            except (json.JSONDecodeError, ValueError):
                pass
            args[key] = val
        tools.append({"name": tool_name, "arguments": args})
    if tools:
        return tools

    # Filter out noise: unknown tool names with empty args (usually XML format examples)
    KNOWN_TOOLS = VALID_TOOLS | {'unknown'}
    tools = [t for t in tools if t['name'] in KNOWN_TOOLS or t['arguments']]
    return tools

def _extract_tool_calls_safe(text: str) -> list:
    """Extract tool calls, but ignore noise (explanatory text with XML examples).
    - Tool call có tên hợp lệ → luôn giữ, kể cả khi có text xung quanh.
    - Tool call fake (noise) → kiểm tra threshold, nếu text thừa quá nhiều thì bỏ.
    """
    tool_calls = _extract_xml_tags(text)
    if not tool_calls:
        return tool_calls

    # Tách tool call hợp lệ vs noise (fake names từ code examples)
    valid = [tc for tc in tool_calls if tc['name'] in VALID_TOOLS]
    noise = [tc for tc in tool_calls if tc['name'] not in VALID_TOOLS]

    # Có tool call hợp lệ → luôn trả về, bất kể text xung quanh
    if valid:
        return valid

    # Toàn noise → kiểm tra threshold để quyết định có phải text giải thích không
    clean = strip_tool_calls(text).strip()
    threshold = max(120, len(text) * 0.35)
    if len(clean) > threshold:
        return []  # text giải thích dài, XML chỉ là ví dụ
    return noise

def strip_tool_calls(text: str) -> str:
    """Remove tool call XML blocks from text."""
    text = re.sub(r'<tool>\s*\w+\s*</tool>\s*<json>.*?</json>', '', text, flags=re.DOTALL)
    text = re.sub(r'<function_call\s+name\s*=\s*"[^"]*"\s*>.*?</function_call>', '', text, flags=re.DOTALL)
    text = re.sub(r'<function_call\s+name\s*=\s*"[^"]*"\s*>\s*<args>.*?</args>', '', text, flags=re.DOTALL)
    text = re.sub(r'<tool>\s*<name>\w+</name>.*?</tool>', '', text, flags=re.DOTALL)
    text = re.sub(r'<tool>\s*<json>.*?</json>\s*</tool>', '', text, flags=re.DOTALL)
    # Format 9: <tool><parameter name="KEY">value</parameter>...</tool>
    text = re.sub(r'<tool>\s*(<parameter\s+name\s*=\s*"\w+"[^>]*>\s*.*?\s*</parameter>\s*)+</tool>', '', text, flags=re.DOTALL)
    # Format 10: <tool><tool_call>NAME</tool_call><json>{...}</json></tool>
    text = re.sub(r'<tool>\s*<tool_call>\w+</tool_call>\s*<json>.*?</json>\s*</tool>', '', text, flags=re.DOTALL)
    # Format 11: <tool><tool>NAME</tool><parameter>k</parameter><parameter>v</parameter>...</tool>
    text = re.sub(r'<tool>\s*<tool>\w+</tool>\s*(<parameter>[^<]*</parameter>\s*)+</tool>', '', text, flags=re.DOTALL)
    # Format 11b: <tool><tool>NAME</tool><parameter>k</parameter><parameter>v</parameter>... (missing </tool>)
    text = re.sub(r'<tool>\s*<tool>\w+</tool>\s*(<parameter>[^<]*</parameter>\s*)+', '', text, flags=re.DOTALL)
    # Format 8: <tool name="TOOL"><parameter ...>...</parameter></tool>
    text = re.sub(r'<tool\s+name\s*=\s*"[^"]*"\s*>.*?</tool>', '', text, flags=re.DOTALL)
    # Format 10b: <tool><tool_call name="NAME"><parameter ...>...</parameter></tool_call></tool>
    text = re.sub(r'<tool>\s*<tool_call\s+name\s*=\s*"\w+"\s*>.*?</tool_call>\s*</tool>', '', text, flags=re.DOTALL)
    # Format 10c: <tool><tool_call>NAME</tool_call><parameter name="KEY">value</parameter>...</tool>
    text = re.sub(r'<tool>\s*<tool_call>\w+</tool_call>\s*(<parameter\s+name\s*=\s*"\w+"[^>]*>\s*.*?\s*</parameter>\s*)+</tool>', '', text, flags=re.DOTALL)
    # Format 12: <tool>{JSON}</tool> (raw JSON)
    text = re.sub(r'<tool>\s*\{[^}]*\}\s*</tool>', '', text, flags=re.DOTALL)
    text = re.sub(r'<tool>\s*\{.*?\}\s*</tool>', '', text, flags=re.DOTALL)
    # Format 13: <tool><tool_name name="X">X</tool_name><json>{...}</json></tool>
    text = re.sub(r'<tool>\s*<tool_name\s+name\s*=\s*"[^"]*"[^>]*>\s*\w+\s*</tool_name>\s*<json>.*?</json>\s*</tool>', '', text, flags=re.DOTALL)
    text = re.sub(r'<tool>\s*<tool_name\s+name\s*=\s*"[^"]*"[^>]*>\s*\w+\s*</tool_name>\s*<json>.*?</json>', '', text, flags=re.DOTALL)
    # Format 16: <tool><invoke name="X"><parameter ...>...</parameter></invoke></tool>
    # (trip: MUST strip before Format 14 to avoid partial stripping)
    text = re.sub(r'<tool>\s*<invoke\s+name\s*=\s*"\w+"\s*>(?:\s*<parameter\s+name\s*=\s*"\w+"[^>]*>\s*.*?\s*</parameter>\s*)+</invoke>\s*</tool>', '', text, flags=re.DOTALL)
    # Format 16b: <tool><invoke name="X"><parameter ...>...</parameter></invoke> (missing </tool>)
    text = re.sub(r'<tool>\s*<invoke\s+name\s*=\s*"\w+"\s*>(?:\s*<parameter\s+name\s*=\s*"\w+"[^>]*>\s*.*?\s*</parameter>\s*)+</invoke>', '', text, flags=re.DOTALL)
    # Format 14: <invoke name="X"><parameter ...>...</parameter></invoke>
    text = re.sub(r'<invoke\s+name\s*=\s*"\w+"\s*>(?:\s*<parameter\s+name\s*=\s*"\w+"[^>]*>\s*.*?\s*</parameter>\s*)+</invoke>', '', text, flags=re.DOTALL)
    # Format 14b: <invoke name="X"><parameter ...>...</parameter> (missing </invoke>)
    text = re.sub(r'<invoke\s+name\s*=\s*"\w+"\s*>(?:\s*<parameter\s+name\s*=\s*"\w+"[^>]*>\s*.*?\s*</parameter>\s*)+', '', text, flags=re.DOTALL)
    # Format 15: <tool><tool_name>X</tool_name><json>...</json></tool>
    text = re.sub(r'<tool>\s*<tool_name>\w+</tool_name>\s*<json>.*?</json>\s*</tool>', '', text, flags=re.DOTALL)
    # Format 15b: <tool><tool_name>X</tool_name><json>...</json> (missing </tool>)
    text = re.sub(r'<tool>\s*<tool_name>\w+</tool_name>\s*<json>.*?</json>', '', text, flags=re.DOTALL)
    # Format 17: <tool><tool_name>X</tool_name><child>v</child>...</tool>
    text = re.sub(r'<tool>\s*<tool_name>\w+</tool_name>(?:\s*<\w+>[^<]*</\w+>\s*)+</tool>', '', text, flags=re.DOTALL)
    # Format 17b: <tool><tool_name>X</tool_name><child>v</child>... (missing </tool>)
    text = re.sub(r'<tool>\s*<tool_name>\w+</tool_name>(?:\s*<\w+>[^<]*</\w+>\s*)+', '', text, flags=re.DOTALL)
    # Format 18: <tool><tool_name>X</tool_name><parameter name="Y">v</parameter>...</tool>
    text = re.sub(r'<tool>\s*<tool_name>\w+</tool_name>\s*(?:<parameter\s+name\s*=\s*"\w+"[^>]*>\s*.*?\s*</parameter>\s*)+</tool>', '', text, flags=re.DOTALL)
    # Format 18b: (missing </tool>)
    text = re.sub(r'<tool>\s*<tool_name>\w+</tool_name>\s*(?:<parameter\s+name\s*=\s*"\w+"[^>]*>\s*.*?\s*</parameter>\s*)+', '', text, flags=re.DOTALL)
    # Format 19: <tool tool_name="X"><json>...</json></tool>
    text = re.sub(r'<tool\s+tool_name\s*=\s*"\w+"\s*>\s*<json>.*?</json>\s*</tool>', '', text, flags=re.DOTALL)
    # Format 19b: (missing </tool>)
    text = re.sub(r'<tool\s+tool_name\s*=\s*"\w+"\s*>\s*<json>.*?</json>', '', text, flags=re.DOTALL)
    # Format 20: <tool><tool_name>X</tool_name><input>...</input></tool>
    text = re.sub(r'<tool>\s*<tool_name>\w+</tool_name>\s*<input>.*?</input>\s*</tool>', '', text, flags=re.DOTALL)
    # Format 20b: (missing </tool>)
    text = re.sub(r'<tool>\s*<tool_name>\w+</tool_name>\s*<input>.*?</input>', '', text, flags=re.DOTALL)
    # Format 21: <tool><tool_name>X</tool_name><arguments>...</arguments></tool>
    text = re.sub(r'<tool>\s*<tool_name>\w+</tool_name>\s*<arguments>.*?</arguments>\s*</tool>', '', text, flags=re.DOTALL)
    # Format 21b: (missing </tool>)
    text = re.sub(r'<tool>\s*<tool_name>\w+</tool_name>\s*<arguments>.*?</arguments>', '', text, flags=re.DOTALL)
    # Format 22: <tool name="X" ... /> (self-closing)
    text = re.sub(r'<tool\s+[^>]+/>', '', text, flags=re.DOTALL)
    # Format 23: <tool tool_name="X" json='...' /> (self-closing)
    text = re.sub(r"<tool\s+tool_name\s*=\s*\"\w+\"\s+json\s*=\s*'[^']*'\s*/>", '', text, flags=re.DOTALL)
    text = re.sub(r'<tool\s+tool_name\s*=\s*"\w+"\s+json\s*=\s*"[^"]*"\s*/>', '', text, flags=re.DOTALL)
    # Format 25: <invoke name="X" ... /> (self-closing)
    text = re.sub(r'<invoke\s+[^>]+/>', '', text, flags=re.DOTALL)
    # Format 26: <tools>...</tools> (multi-tool wrapper)
    text = re.sub(r'<tools>.*?</tools>', '', text, flags=re.DOTALL)
    tool_names = sorted(VALID_TOOLS)  # sync với VALID_TOOLS
    IGN2 = re.DOTALL | re.IGNORECASE
    for tname in tool_names:
        text = re.sub(rf'<tool>\s*<{tname}>.*?</{tname}>\s*</tool>', '', text, flags=IGN2)
        text = re.sub(rf'<tool>\s*<{tname}>\s*<json>.*?</json>\s*</tool>', '', text, flags=IGN2)
        text = re.sub(rf'<tool>\s*<{tname}>.*?</tool>', '', text, flags=IGN2)
        text = re.sub(rf'<{tname}>.*?</{tname}>', '', text, flags=IGN2)
    return text.strip()

def _get_valid_tool_set(tools_param=None):
    """Extract tool names from tools parameter (OpenAI format).
    Nếu client gửi tools → validate theo danh sách đó.
    Nếu không → dùng VALID_TOOLS mặc định.
    """
    if not tools_param:
        return set(VALID_TOOLS)
    valid = set(VALID_TOOLS)  # base set luôn được chấp nhận
    for t in tools_param:
        func = t.get('function', {})
        name = func.get('name', '')
        if name:
            valid.add(name)
    return valid

def _validate_tool_calls(tool_calls, valid_set=None):
    """Validate tool calls (case-insensitive). Returns (valid_calls, error_message).
    Nếu có tool name không hợp lệ → trả về list rỗng + thông báo lỗi
    để model biết và tự sửa.
    """
    if not tool_calls:
        return [], None
    if valid_set is None:
        valid_set = VALID_TOOLS
    # Case-insensitive matching: askuserquestion ↔ AskUserQuestion
    valid_lower = {name.lower(): name for name in valid_set}
    unknown = [tc['name'] for tc in tool_calls if tc['name'].lower() not in valid_lower]
    valid = []
    for tc in tool_calls:
        canonical = valid_lower.get(tc['name'].lower())
        if canonical:
            valid.append({"name": canonical, "arguments": tc["arguments"]})
    if unknown and not valid:
        # Tất cả tool đều sai → báo lỗi để model sửa
        msg = (
            f"TOOL CALL ERROR: Unknown tool(s): {', '.join(unknown)}. "
            f"Valid tools: {', '.join(sorted(valid_set))}. "
            f"Please correct and use only valid tools from the list."
        )
        return [], msg
    if unknown:
        # Có tool sai lẫn tool đúng → giữ tool đúng, log cảnh báo
        print(f"[validate] Dropped unknown tools: {unknown}, kept: {[t['name'] for t in valid]}", flush=True)
    return valid, None

