import numpy as np
import struct
from scipy.fftpack import idct

def read_jpeg(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()
    # 解析JPEG头部
    header_info = parse_jpeg_header(data)
    # 提取霍夫曼表
    huffman_tables = parse_huffman_tables(data)
    # 提取量化表
    quant_tables = parse_quantization_tables(data)
    # 提取压缩数据
    compressed_data = extract_compressed_data(data)
    return data, header_info, huffman_tables, quant_tables, compressed_data

def parse_jpeg_header(data):
    if data[0:2] != b'\xff\xd8':
        raise ValueError("不是有效的JPEG文件。")

    index = 2
    while index < len(data):
        # 检查是否到达数据末尾
        if index + 2 > len(data):
            raise ValueError("文件数据不完整。")
            
        # 查找标记的开始
        if data[index] != 0xFF:
            index += 1
            continue
            
        # 跳过填充字节
        while index < len(data) and data[index] == 0xFF:
            index += 1
            
        if index >= len(data):
            raise ValueError("文件数据不完整。")
            
        marker = data[index]
        index += 1
        
        if marker == 0xC0:  # SOF0
            if index + 8 > len(data):
                raise ValueError("SOF0 数据不完整。")
            length = struct.unpack('>H', data[index:index+2])[0]
            bits = data[index+2]
            height = struct.unpack('>H', data[index+3:index+5])[0]
            width = struct.unpack('>H', data[index+5:index+7])[0]
            components = data[index+7]
            return {
                'width': width,
                'height': height,
                'components': components,
                'bits': bits,
                'color_space': 'YCbCr' if components == 3 else 'Grayscale'
            }
        else:
            # 跳过其他标记
            if index + 2 > len(data):
                raise ValueError("标记长度数据不完整。")
            length = struct.unpack('>H', data[index:index+2])[0]
            index += length
            
    raise ValueError("未找到SOF0标记。")

def extract_compressed_data(data):
    import struct

    index = 2
    while index < len(data):
        # 检查是否到达数据末尾
        if index + 2 > len(data):
            break
            
        # 查找标记的开始
        if data[index] != 0xFF:
            index += 1
            continue
            
        # 跳过填充字节
        while index < len(data) and data[index] == 0xFF:
            index += 1
            
        if index >= len(data):
            break
            
        marker = data[index]
        index += 1
        
        if marker == 0xDA:  # SOS (Start of Scan)
            if index + 2 > len(data):
                raise ValueError("SOS 长度数据不完整。")
                
            length = struct.unpack('>H', data[index:index+2])[0]
            index += 2
            scan_data_start = index + length - 2
            
            # 查找EOI标记 (0xFFD9)，考虑填充字节
            scan_data_end = -1
            i = scan_data_start
            while i < len(data) - 1:
                if data[i] == 0xFF and data[i + 1] == 0xD9:
                    scan_data_end = i
                    break
                i += 1
                
            if scan_data_end == -1:
                raise ValueError("未找到EOI标记。")
                
            # 提取压缩数据
            compressed_data = bytearray()
            i = scan_data_start
            while i < scan_data_end:
                byte = data[i]
                compressed_data.append(byte)
                # 如果遇到0xFF，检查下一个字节
                if byte == 0xFF:
                    i += 1
                    if i < scan_data_end:
                        next_byte = data[i]
                        # 如果下一个字节是0x00，这是填充，跳过它
                        if next_byte == 0x00:
                            i += 1
                            continue
                        # 如果是其他值，可能是标记
                        elif next_byte != 0xFF:
                            compressed_data.append(next_byte)
                i += 1
                
            return bytes(compressed_data)
        else:
            # 跳过其他标记
            if index + 2 > len(data):
                break
            length = struct.unpack('>H', data[index:index+2])[0]
            index += length
            
    raise ValueError("未找到SOS标记。")

def build_huffman_table(code_lengths, symbols):
    # 使用 NumPy 进行计算
    total_codes = np.sum(code_lengths)
    if total_codes != len(symbols):
        print(f"错误: code_lengths 的总和 ({total_codes}) 与 symbols 的长度 ({len(symbols)}) 不匹配。")
        raise ValueError("code_lengths 与 symbols 长度不一致。")
    
    huffman_table = {}
    code = 0
    k = 0
    
    # 预分配数组以提高性能
    codes = np.zeros(total_codes, dtype=np.uint32)
    lengths = np.zeros(total_codes, dtype=np.uint8)
    
    for i in range(16):
        code <<= 1
        n_codes = code_lengths[i]
        if n_codes > 0:
            codes[k:k+n_codes] = np.arange(code, code + n_codes)
            lengths[k:k+n_codes] = i + 1
            code += n_codes
            k += n_codes
    
    # 一次性创建霍夫曼表
    for i in range(total_codes):
        huffman_table[(int(codes[i]), int(lengths[i]))] = symbols[i]
    
    return huffman_table

def parse_huffman_tables(data):
    import struct
    from collections import defaultdict
    import numpy as np

    huffman_tables = {'DC': {}, 'AC': {}}
    index = 2
    
    # 使用 memoryview 来避免数据复制
    data_view = memoryview(data)
    
    while index < len(data):
        # 检查是否到达数据末尾
        if index + 2 > len(data):
            break
            
        # 查找标记的开始
        if data[index] != 0xFF:
            index += 1
            continue
            
        # 跳过填充字节
        while index < len(data) and data[index] == 0xFF:
            index += 1
            
        if index >= len(data):
            break
            
        marker = data[index]
        index += 1
        
        if marker == 0xC4:  # DHT
            if index + 2 > len(data):
                raise ValueError("DHT 长度数据不完整。")
                
            length = struct.unpack('>H', data[index:index+2])[0]
            index += 2
            
            # 检查DHT数据块的完整性
            if index + length - 2 > len(data):
                raise ValueError("DHT 数据块不完整。")
                
            dht_end = index + length - 2
            dht_data = data_view[index:dht_end]
            
            pos = 0
            while pos < len(dht_data):
                # 读取表信息
                if pos + 17 > len(dht_data):  # 1字节表信息 + 16字节码长
                    break
                    
                ht_info = dht_data[pos]
                table_class = (ht_info >> 4) & 0x0F  # 0 = DC, 1 = AC
                table_id = ht_info & 0x0F
                pos += 1
                
                # 读取码长
                code_lengths = np.frombuffer(dht_data[pos:pos+16], dtype=np.uint8).copy()
                pos += 16
                
                # 计算符号总数
                total_symbols = int(np.sum(code_lengths))
                
                # 检查符号数据的完整性
                if pos + total_symbols > len(dht_data):
                    break
                    
                # 读取符号
                symbols = np.frombuffer(dht_data[pos:pos+total_symbols], dtype=np.uint8).copy()
                pos += total_symbols
                
                # 构建霍夫曼表
                table_type = 'DC' if table_class == 0 else 'AC'
                try:
                    huffman_table = build_huffman_table(code_lengths, symbols)
                    huffman_tables[table_type][table_id] = huffman_table
                except (ValueError, IndexError) as e:
                    print(f"警告: 解析霍夫曼表时出错: {e}")
                    continue
            
            index = dht_end + 2
            
        elif marker == 0xDA:  # SOS
            break
            
        else:
            # 跳过其他标记
            if index + 2 > len(data):
                break
            length = struct.unpack('>H', data[index:index+2])[0]
            index += length
            
    if not huffman_tables['DC'] or not huffman_tables['AC']:
        raise ValueError("未找到必要的霍夫曼表。")
        
    return huffman_tables

def parse_quantization_tables(data):
    import struct
    import numpy as np

    quantization_tables = {}
    index = 2
    
    while index < len(data):
        # 检查是否到达数据末尾
        if index + 2 > len(data):
            break
            
        # 查找标记的开始
        if data[index] != 0xFF:
            index += 1
            continue
            
        # 跳过填充字节
        while index < len(data) and data[index] == 0xFF:
            index += 1
            
        if index >= len(data):
            break
            
        marker = data[index]
        index += 1
        
        if marker == 0xDB:  # DQT
            if index + 2 > len(data):
                break
                
            length = struct.unpack('>H', data[index:index+2])[0]
            index += 2
            
            # 检查DQT数据块的完整性
            if index + length - 2 > len(data):
                break
                
            dqt_end = index + length - 2
            
            while index < dqt_end:
                if index + 1 > len(data):
                    break
                    
                pq_tq = data[index]
                pq = (pq_tq >> 4) & 0x0F  # Precision: 0 = 8-bit, 1 = 16-bit
                tq = pq_tq & 0x0F          # Table identifier
                index += 1
                
                table_size = 128 if pq == 1 else 64
                if index + table_size > len(data):
                    break
                    
                if pq == 0:
                    q_table = np.array([data[index+i] for i in range(64)], dtype=np.uint8).reshape((8,8))
                    index += 64
                elif pq == 1:
                    q_table = np.array([(data[index+2*i] << 8) | data[index+2*i+1] 
                                      for i in range(64)], dtype=np.uint16).reshape((8,8))
                    index += 128
                else:
                    index = dqt_end
                    continue
                
                quantization_tables[tq] = q_table
            
            index = dqt_end + 2
            
        elif marker == 0xDA:  # SOS
            break
            
        else:
            # 跳过其他标记
            if index + 2 > len(data):
                break
            length = struct.unpack('>H', data[index:index+2])[0]
            index += length
            
    if not quantization_tables:
        raise ValueError("未找到任何量化表。")
    
    return quantization_tables

def huffman_decode(bitstream, huffman_tables):
    decoded_symbols = []
    
    # 假设使用表ID 0 的 DC 和 AC 表
    dc_table = huffman_tables['DC'].get(0, {})
    ac_table = huffman_tables['AC'].get(0, {})
    
    # 预计算最大的霍夫曼码长度
    max_dc_length = max((length for (code, length) in dc_table.keys()), default=0)
    max_ac_length = max((length for (code, length) in ac_table.keys()), default=0)
    max_code_length = max(max_dc_length, max_ac_length)
    
    total_bits = len(bitstream)
    last_percentage = 0
    i = 0  # 当前比特位置

    while i < total_bits:
        matched = False
        # 尝试匹配 DC 表
        for length in range(1, max_dc_length + 1):
            if i + length > total_bits:
                break
            # 提取当前长度的码
            code = 0
            for bit in bitstream[i:i+length]:
                code = (code << 1) | bit
            symbol = dc_table.get((code, length))
            if symbol is not None:
                decoded_symbols.append(('DC', symbol))
                i += length
                matched = True
                break
        if matched:
            # 更新进度
            percentage = int((i) / total_bits * 100)
            if percentage > last_percentage:
                print(f"解码进度: {percentage}%")
                last_percentage = percentage
            continue

        # 尝试匹配 AC 表
        for length in range(1, max_ac_length + 1):
            if i + length > total_bits:
                break
            code = 0
            for bit in bitstream[i:i+length]:
                code = (code << 1) | bit
            symbol = ac_table.get((code, length))
            if symbol is not None:
                decoded_symbols.append(('AC', symbol))
                i += length
                matched = True
                break
        if matched:
            # 更新进度
            percentage = int((i) / total_bits * 100)
            if percentage > last_percentage:
                print(f"解码进度: {percentage}%")
                last_percentage = percentage
            continue

        # 如果没有匹配到任何表，则跳过一个比特（错误处理）
        i += 1

    return decoded_symbols

def read_bitstream(compressed_data):
    # 使用位操作而非字符串拼接
    bitstream = []
    for byte in compressed_data:
        bits = [(byte >> bit) & 1 for bit in reversed(range(8))]
        bitstream.extend(bits)
    return bitstream

def reconstruct_dct(decoded_symbols, huffman_tables, header_info):
    """
    根据解码后的符号重建DCT系数块。
    这里只是一个简化的示例，实际实现需要处理更多细节。
    """
    dct_coefficients = []
    current_block = np.zeros((8,8), dtype=np.int32)
    zz_order = [
        0, 1, 5, 6,14,15,27,28,
        2,4,7,13,16,26,29,42,
        3,8,12,17,25,30,41,43,
        9,11,18,24,31,40,44,53,
        10,19,23,32,39,45,52,54,
        20,22,33,38,46,51,55,60,
        21,34,37,47,50,56,59,61,
        35,36,48,49,57,58,62,63
    ]
    prev_dc = 0
    index = 0

    for symbol_type, symbol in decoded_symbols:
        if symbol_type == 'DC':
            # DC符号代表差分值，需要将其与前一个DC值相加
            dc_diff = symbol
            dc_value = prev_dc + dc_diff
            current_block[0,0] = dc_value
            prev_dc = dc_value
        elif symbol_type == 'AC':
            # AC符号采用运行长度编码
            run = (symbol >> 4) & 0x0F
            size = symbol & 0x0F
            if run == 0 and size == 0:
                # EOB (End of Block)
                dct_coefficients.append(current_block.copy())
                current_block[:, :] = 0
                continue
            # 在Z字形顺序中跳过run个1
            pos = 1
            while run > 0 and pos < 64:
                current_block.flat[zz_order[pos]] = 0
                run -= 1
                pos += 1
            if pos >= 64:
                break
            # 读取接下来的size位来获取具体的AC值（这里简化为直接赋值）
            # 实际上需要根据size读取更多位，并进行扩展
            ac_value = symbol  # 这只是一个占位符
            current_block.flat[zz_order[pos]] = ac_value
        index += 1

    if not np.all(current_block == 0):
        dct_coefficients.append(current_block.copy())

    return dct_coefficients

def inverse_quantize(dct_coefficients, quant_tables):
    """
    反量化处理
    dct_coefficients: list of 8x8 numpy arrays
    quant_tables: dict mapping table identifier to 8x8 numpy arrays
    """
    inverse_quantized_blocks = []
    for block in dct_coefficients:
        # 假设所有块都使用相同的量化表ID 0
        q_table = quant_tables.get(0)
        if q_table is None:
            raise ValueError("未找到量化表ID 0。")
        inverse_block = block * q_table
        inverse_quantized_blocks.append(inverse_block)
    return inverse_quantized_blocks

def perform_idct(inverse_quantized_blocks):
    idct_blocks = []
    for block in inverse_quantized_blocks:
        # 执行二维IDCT
        idct_block = idct(idct(block.T, norm='ortho').T, norm='ortho')
        idct_blocks.append(idct_block)
    return idct_blocks

def color_space_conversion(idct_blocks, header_info):
    """
    从YCbCr转换到RGB
    """
    height = header_info['height']
    width = header_info['width']
    components = header_info['components']
    
    if components != 3:
        raise NotImplementedError("仅支持YCbCr颜色空间的JPEG图像。")
    
    # 假设每个块对应8x8像素
    blocks_per_row = width // 8
    blocks_per_col = height // 8
    
    # 初始化Y, Cb, Cr数组
    Y = np.zeros((height, width), dtype=np.float32)
    Cb = np.zeros((height, width), dtype=np.float32)
    Cr = np.zeros((height, width), dtype=np.float32)
    
    block_index = 0
    for i in range(blocks_per_col):
        for j in range(blocks_per_row):
            if block_index >= len(idct_blocks):
                break
            block = idct_blocks[block_index]
            block_index += 1
            # 将Z字形顺序转为行列顺序
            zz_order = [
                0, 1, 5, 6,14,15,27,28,
                2,4,7,13,16,26,29,42,
                3,8,12,17,25,30,41,43,
                9,11,18,24,31,40,44,53,
                10,19,23,32,39,45,52,54,
                20,22,33,38,46,51,55,60,
                21,34,37,47,50,56,59,61,
                35,36,48,49,57,58,62,63
            ]
            block_ordered = np.array([block.flat[zz_order[k]] for k in range(64)]).reshape((8,8))
            Y_block = block_ordered
            Cb_block = block_ordered
            Cr_block = block_ordered
            Y[i*8:(i+1)*8, j*8:(j+1)*8] = Y_block
            Cb[i*8:(i+1)*8, j*8:(j+1)*8] = Cb_block
            Cr[i*8:(i+1)*8, j*8:(j+1)*8] = Cr_block
    
    # YCbCr to RGB conversion
    R = Y + 1.402 * (Cr - 128)
    G = Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)
    B = Y + 1.772 * (Cb - 128)
    
    # Clip the values to [0,255]
    R = np.clip(R, 0, 255).astype(np.uint8)
    G = np.clip(G, 0, 255).astype(np.uint8)
    B = np.clip(B, 0, 255).astype(np.uint8)
    
    # 合并R, G, B通道
    rgb_image = np.dstack((R, G, B))
    
    return rgb_image

def save_as_bmp(rgb_image, output_path):
    """
    将RGB图像数据保存为BMP文件
    """
    height, width, _ = rgb_image.shape
    # BMP文件头
    file_type = b'BM'
    file_size = 14 + 40 + width * height * 3
    reserved_1 = 0
    reserved_2 = 0
    offset = 14 + 40
    bmp_header = struct.pack('<2sIHHI', file_type, file_size, reserved_1, reserved_2, offset)
    
    # DIB头（BITMAPINFOHEADER）
    dib_header_size = 40
    planes = 1
    bits_per_pixel = 24
    compression = 0
    image_size = width * height * 3
    x_pixels_per_meter = 0
    y_pixels_per_meter = 0
    total_colors = 0
    important_colors = 0
    dib_header = struct.pack('<IIIHHIIIIII',
                             dib_header_size,
                             width,
                             height,
                             planes,
                             bits_per_pixel,
                             compression,
                             image_size,
                             x_pixels_per_meter,
                             y_pixels_per_meter,
                             total_colors,
                             important_colors)
    
    with open(output_path, 'wb') as f:
        f.write(bmp_header)
        f.write(dib_header)
        # BMP文件从下到上写
        for row in reversed(rgb_image):
            f.write(row.tobytes())
            # 每行字节数必须是4的倍数，填充0
            padding = (4 - (width * 3) % 4) % 4
            f.write(b'\x00' * padding)

def main():
    file_path = "input.jpg"  # 请将此路径替换为实际的JPEG文件路径
    try:
        print("开始读取JPEG文件...")
        data, header_info, huffman_tables, quant_tables, compressed_data = read_jpeg(file_path)
        print("成功读取JPEG文件")
        
        print("\nJPEG头部信息：")
        print(f"宽度: {header_info['width']} 像素")
        print(f"高度: {header_info['height']} 像素")
        print(f"颜色空间: {header_info['color_space']}")
        print(f"颜色组件: {header_info['components']}")
        print(f"每个颜色组件的位数: {header_info['bits']}")
        print(f"压缩数据长度: {len(compressed_data)} 字节")
        
        print("\n开始读取比特流...")
        bitstream = read_bitstream(compressed_data)
        print("成功读取比特流")
        
        print("\n开始霍夫曼解码...")
        decoded_symbols = huffman_decode(bitstream, huffman_tables)
        print(f"解码完成，解码后的符号数量: {len(decoded_symbols)}")
        
        print("\n开始重建DCT系数块...")
        dct_coefficients = reconstruct_dct(decoded_symbols, huffman_tables, header_info)
        print("成功重建DCT系数块")
        
        print("\n开始反量化...")
        inverse_quantized = inverse_quantize(dct_coefficients, quant_tables)
        print("反量化完成")
        
        print("\n开始执行IDCT...")
        idct_blocks = perform_idct(inverse_quantized)
        print("IDCT完成")
        
        print("\n开始色彩空间转换...")
        rgb_image = color_space_conversion(idct_blocks, header_info)
        print("色彩空间转换完成")
        
        print("\n开始保存BMP文件...")
        save_as_bmp(rgb_image, "output.bmp")
        print("BMP文件已保存为 output.bmp")
        
    except ValueError as e:
        print(f"错误: {e}")
    except IndexError as e:
        print(f"错误: 索引越界: {e}")
    except Exception as e:
        print(f"未知错误: {e}")

if __name__ == "__main__":
    main()


