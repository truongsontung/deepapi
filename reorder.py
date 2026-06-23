#!/usr/bin/env python3
import re
from pathlib import Path

# Đọc file gốc
content = Path('/home/vps2/deepapi/tool_parser.py').read_text()

# Tìm vị trí hàm _extract_xml_tags
start = content.find('def _extract_xml_tags(text: str) -> list:')
if start == -1:
    raise ValueError("Không tìm thấy hàm _extract_xml_tags")

end = content.find('def _extract_tool_calls_safe', start)
if end == -1:
    end = content.find('def strip_tool_calls', start)
if end == -1:
    end = content.find('def _get_valid_tool_set', start)
if end == -1:
    raise ValueError("Không tìm thấy kết thúc hàm")

before = content[:start]
after = content[end:]

# Lấy phần thân hàm
text_body = content[start:end]
lines = text_body.splitlines(True)  # giữ newline

# Xác định các khối format bằng comment "# Format"
format_blocks = []
i = 0
while i < len(lines):
    line = lines[i]
    stripped = line.strip()
    if stripped.startswith('# Format'):
        # Lấy số thứ tự
        match = re.search(r'# Format\s*(\d+[a-z]*)', line)
        if match:
            num = match.group(1)
            start_idx = i
            # Tìm kết thúc khối: dòng tiếp theo bắt đầu bằng '    # Format' hoặc dòng không thụt lề (và không phải comment)
            j = i + 1
            while j < len(lines):
                if lines[j].strip().startswith('# Format'):
                    break
                if not lines[j].startswith('    ') and lines[j].strip() != '' and not lines[j].startswith('#'):
                    # Nếu dòng không thụt lề và không phải comment, có thể là kết thúc khối
                    # Nhưng cần kiểm tra xem có phải là dòng '    if tools:' không (thường xuất hiện sau mỗi khối)
                    # Thay vào đó, tôi sẽ kiểm tra nếu dòng bắt đầu bằng '    ' nhưng không phải comment thì vẫn trong khối
                    # Vì vậy tôi sẽ tiếp tục cho đến khi gặp một dòng không thụt lề và không phải là comment '# Format'
                    # Nhưng các dòng trong khối đều bắt đầu bằng '    ' hoặc comment '# Format'
                    # Nên nếu gặp dòng không thụt lề thì đó là kết thúc
                    break
                j += 1
            end_idx = j
            format_blocks.append((num, start_idx, end_idx))
            i = j
        else:
            i += 1
    else:
        i += 1

# Thứ tự ưu tiên mong muốn
priority = [
    '26',
    '1', '1c',
    '2', '2b',
    '3',
    '6', '7',
    '5', '5b',
    '4',
    '8',
    '10', '10b', '10c',
    '11', '11b',
    '12',
    '13', '13b',
    '14', '14b',
    '15', '15b',
    '16', '16b',
    '17', '17b',
    '18', '18b',
    '19', '19b',
    '20', '20b',
    '21', '21b',
    '22', '23', '24', '25',
    '1b', '9',
    '27', '27b'
]

# Hàm sắp xếp key
def sort_key(num):
    match = re.match(r'(\d+)([a-z]*)', num)
    if match:
        n = int(match.group(1))
        suffix = match.group(2)
        return (n, suffix)
    return (999, num)

# Sắp xếp các khối theo thứ tự ưu tiên
ordered_blocks = []
for num in priority:
    found = None
    for block in format_blocks:
        if block[0] == num:
            found = block
            break
    if found:
        ordered_blocks.append(found)
    else:
        print(f"Warning: Không tìm thấy format {num}")

# Tìm vị trí dòng '    tools = []'
tools_line = None
for idx, line in enumerate(lines):
    if line.strip() == 'tools = []':
        tools_line = idx
        break

# Tìm dòng '    return tools'
return_line = None
for idx, line in enumerate(lines):
    if line.strip().startswith('return tools'):
        return_line = idx
        break

if tools_line is None or return_line is None:
    raise ValueError("Không tìm thấy tools = [] hoặc return tools")

# Tạo header: từ đầu đến dòng tools = [] (bao gồm)
header = ''.join(lines[:tools_line+1])
# Footer: từ dòng return tools đến cuối
footer = ''.join(lines[return_line:])

# Xây dựng thân hàm mới
new_body = header
for num, start_idx, end_idx in ordered_blocks:
    block = ''.join(lines[start_idx:end_idx])
    new_body += block
    new_body += '\n'  # thêm dòng trống

new_body += footer

# Ghép lại toàn bộ file
new_content = before + new_body + after

# Ghi file mới
output_path = Path('/home/vps2/deepapi/tool_parser_dev.py')
output_path.write_text(new_content)
print(f"Đã tạo {output_path}")
