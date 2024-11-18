import struct
import math
import os
import numpy as np
from typing import BinaryIO, List, Dict, Tuple

class JPEGDecoder:
    def __init__(self):
        self.quantization_tables = {}
        self.huffman_tables = {'dc': {}, 'ac': {}}
        self.height = 0
        self.width = 0
        self.components = []
        self.bits_data = []
        self.dc_pred = [0] * 3  # DC预测值
        
        # 添加zigzag顺序定义
        self.zigzag_order = [
            0,  1,  5,  6,  14, 15, 27, 28,
            2,  4,  7,  13, 16, 26, 29, 42,
            3,  8,  12, 17, 25, 30, 41, 43,
            9,  11, 18, 24, 31, 40, 44, 53,
            10, 19, 23, 32, 39, 45, 52, 54,
            20, 22, 33, 38, 46, 51, 55, 60,
            21, 34, 37, 47, 50, 56, 59, 61,
            35, 36, 48, 49, 57, 58, 62, 63
        ]

    def read_word(self, file: BinaryIO) -> int:
        """读取2字节"""
        try:
            data = file.read(2)
            if len(data) != 2:
                raise IOError("无法读取足够的数据，文件可能已损坏或不是有效的JPEG文件")
            return struct.unpack('>H', data)[0]
        except struct.error:
            raise IOError("无法解析文件数据，文件可能已损坏或不是有效的JPEG文件")
    
    def read_byte(self, file: BinaryIO) -> int:
        """读取1字节"""
        try:
            data = file.read(1)
            if len(data) != 1:
                raise IOError("无法读取足够的数据，文件可能已损坏或不是有效的JPEG文件")
            return struct.unpack('B', data)[0]
        except struct.error:
            raise IOError("无法解析文件数据，文件可能已损坏或不是有效的JPEG文件")
    
    def read_markers(self, file: BinaryIO) -> None:
        """读取JPEG标记段"""
        try:
            # 检查文件开始标记
            start_marker = self.read_word(file)
            if start_marker != 0xFFD8:
                raise IOError(f"无效的JPEG文件：文件开始标记错误 (0x{start_marker:04X})")
            print("检测到JPEG文件头")
            
            while True:
                # 查找下一个标记
                while True:
                    byte = file.read(1)
                    if not byte:
                        raise IOError("文件意外结束")
                    if byte[0] == 0xFF:
                        next_byte = file.read(1)
                        if not next_byte:
                            raise IOError("文件意外结束")
                        if next_byte[0] != 0x00:  # 不是填充字节
                            marker = (0xFF << 8) | next_byte[0]
                            break
                
                print(f"读取到标记：0x{marker:04X}")
                
                # 处理标记
                if marker == 0xFFD9:  # EOI标记
                    print("读取到文件结束标记")
                    break
                elif marker == 0xFFC0:  # SOF0标记
                    length = self.read_word(file) - 2
                    self._read_sof0(file)
                elif marker == 0xFFDB:  # DQT标记
                    length = self.read_word(file) - 2
                    self._read_quantization_table(file)
                elif marker == 0xFFC4:  # DHT标记
                    length = self.read_word(file) - 2
                    self._read_huffman_table(file)
                elif marker == 0xFFDA:  # SOS标记
                    length = self.read_word(file) - 2
                    self._read_scan_data(file)
                    break  # SOS之后是压缩数据
                elif (marker & 0xFF00) == 0xFF00:  # 其他JPEG标记
                    try:
                        length = self.read_word(file) - 2
                        print(f"跳过标记 0x{marker:04X}, 长度 {length}")
                        file.seek(length, 1)  # 跳过不要的段
                    except Exception as e:
                        print(f"警告：跳过标记时出错 (0x{marker:04X}): {str(e)}")
                        # 尝试继续读取下一个标记
                        continue
                else:
                    print(f"警告：遇到无效标记 0x{marker:04X}，尝试重新同步")
                    # 尝试重新同步到下一个有效标记
                    continue
                
        except Exception as e:
            raise IOError(f"读取JPEG标记段时出错：{str(e)}")

    def _read_sof0(self, file: BinaryIO) -> None:
        """读取图像基本信息"""
        precision = self.read_byte(file)
        self.height = self.read_word(file)
        self.width = self.read_word(file)
        components_count = self.read_byte(file)
        
        print(f"图像信息: {self.width}x{self.height}, {components_count}个颜色分量, {precision}位精度")
        
        for _ in range(components_count):
            component_id = self.read_byte(file)
            sampling_factors = self.read_byte(file)
            qt_table_id = self.read_byte(file)
            
            h_sampling = (sampling_factors >> 4) & 0x0F
            v_sampling = sampling_factors & 0x0F
            
            self.components.append({
                'id': component_id,
                'h_sampling': h_sampling,
                'v_sampling': v_sampling,
                'qt_table_id': qt_table_id
            })
            print(f"分量 {component_id}: 采样因子 {h_sampling}x{v_sampling}, 量化表 {qt_table_id}")

    def _read_quantization_table(self, file: BinaryIO) -> None:
        """读取量化表"""
        length = self.read_word(file) - 2
        while length > 0:
            table_info = self.read_byte(file)
            table_id = table_info & 0x0F
            precision = (table_info >> 4) & 0x0F
            
            table = []
            for _ in range(64):
                if precision == 0:
                    table.append(self.read_byte(file))
                else:
                    table.append(self.read_word(file))
            
            self.quantization_tables[table_id] = table
            length -= 65 if precision == 0 else 129
            print(f"读取量化表 {table_id}")

    def _build_huffman_table(self, bits_length: List[int], huffman_codes: List[int]) -> Dict:
        """构建霍夫曼表"""
        huffman_table = {}
        code = 0
        pos = 0
        
        print("\n构建霍夫曼表:")
        print(f"位长度数组: {bits_length}")
        print(f"编码数组: {huffman_codes[:10]}...")
        
        for bits in range(1, 17):
            for i in range(bits_length[bits - 1]):
                if pos >= len(huffman_codes):
                    break
                binary = format(code, f'0{bits}b')
                huffman_table[binary] = huffman_codes[pos]
                print(f"添加编码: {binary} -> {huffman_codes[pos]}")
                pos += 1
                code += 1
            code <<= 1
        
        return huffman_table

    def _read_huffman_table(self, file: BinaryIO) -> None:
        """读取霍夫曼表"""
        try:
            length = self.read_word(file) - 2
            bytes_read = 0
            
            while bytes_read < length:
                table_info = self.read_byte(file)
                table_id = table_info & 0x0F
                is_ac = (table_info >> 4) & 0x01
                
                print(f"\n读取{'AC' if is_ac else 'DC'}霍夫曼表 {table_id}")
                
                # 读取位长度计数
                bits_length = []
                total_symbols = 0
                for i in range(16):
                    count = self.read_byte(file)
                    bits_length.append(count)
                    total_symbols += count
                
                print(f"位长度统计: {bits_length}")
                print(f"总符号数: {total_symbols}")
                
                # 读取霍夫曼编码对应的符号
                huffman_codes = []
                for i in range(total_symbols):
                    symbol = self.read_byte(file)
                    huffman_codes.append(symbol)
                
                print(f"符号列表: {huffman_codes[:10]}...")
                
                # 构建霍夫曼表
                table_type = 'ac' if is_ac else 'dc'
                self.huffman_tables[table_type][table_id] = self._build_huffman_table(bits_length, huffman_codes)
                
                bytes_read += 1 + 16 + total_symbols
                
        except Exception as e:
            raise IOError(f"读取霍夫曼表时出错：{str(e)}")

    def _read_scan_data(self, file: BinaryIO) -> None:
        """读取扫描数据"""
        try:
            length = self.read_word(file) - 2
            components_in_scan = self.read_byte(file)
            
            # 读取每个颜色分量使用的霍夫曼表
            scan_components = []
            for _ in range(components_in_scan):
                component_id = self.read_byte(file)
                huffman_table_ids = self.read_byte(file)
                scan_components.append({
                    'id': component_id,
                    'dc_table_id': (huffman_table_ids >> 4) & 0x0F,
                    'ac_table_id': huffman_table_ids & 0x0F
                })
            
            # 跳过3个字节（包含开始谱写位置、结束谱写位置和连续近似位置）
            file.read(3)
            
            # 读取压缩数据
            self._read_entropy_coded_data(file)
            print("成功读取扫描数据")
            
        except Exception as e:
            raise IOError(f"读取扫描数据时出错：{str(e)}")

    def _read_entropy_coded_data(self, file: BinaryIO) -> None:
        """读取熵编码数据"""
        try:
            compressed_data = []
            prev_byte = 0x00
            
            while True:
                current_byte = self.read_byte(file)
                
                if prev_byte == 0xFF:
                    if current_byte == 0x00:
                        compressed_data.append(0xFF)
                        prev_byte = 0x00
                        continue
                    elif current_byte == 0xD9:  # EOI标记
                        break
                    else:
                        raise IOError(f"在压缩数据中遇到意外的标记：0xFF{current_byte:02X}")
                
                compressed_data.append(current_byte)
                prev_byte = current_byte
            
            # 将字节数据转换为位流
            bits = []
            for byte in compressed_data:
                for i in range(7, -1, -1):  # 从最高位到最低位
                    bits.append((byte >> i) & 1)
            self.bits_data = np.array(bits, dtype=np.int32)
            
            print(f"读取到 {len(compressed_data)} 字节的压缩数据")
            print(f"转换为 {len(bits)} 位的位流")
            
        except Exception as e:
            raise IOError(f"读取熵编码数据时出错：{str(e)}")

    def _get_next_bits(self, n: int) -> str:
        """从位流中获取下n位"""
        if not self.bits_data.size:
            raise IOError("位流数据不足")
        if self.bits_data.size < n:
            raise IOError(f"请求{n}位但只剩{self.bits_data.size}位")
        
        result = ''.join(str(bit) for bit in self.bits_data[:n])
        self.bits_data = self.bits_data[n:]
        return result

    def decode(self, jpeg_path: str, bmp_path: str) -> None:
        """解码JPEG文件并保存为BMP"""
        try:
            # 检查文件是否存在
            if not os.path.exists(jpeg_path):
                raise FileNotFoundError(f"找不到输入文件：{jpeg_path}")
            
            # 检查文件大小
            if os.path.getsize(jpeg_path) == 0:
                raise IOError("输入文件是空文件")
            
            # 读取JPEG文件
            with open(jpeg_path, 'rb') as f:
                try:
                    self.read_markers(f)
                except IOError as e:
                    raise IOError(f"读取JPEG文件时出错：{str(e)}")
                
            # 准备扫描组件信息
            scan_components = []
            for component in self.components:
                scan_components.append({
                    'id': component['id'],
                    'dc_table_id': 0 if component['id'] == 1 else 1,  # 通常Y使用表0，Cb/Cr使用表1
                    'ac_table_id': 0 if component['id'] == 1 else 1
                })
            
            print("开始解码霍夫曼数据...")
            decoded_blocks = self._decode_huffman_data(scan_components)
            print("霍夫曼解码完成")
            
            print("重组解码后的数据块...")
            # 码后的数据按颜色分量组织
            mcu_rows = (self.height + 7) // 8
            mcu_cols = (self.width + 7) // 8
            
            # 为每个颜色分量创建数据数组
            component_data = {}
            for component in self.components:
                h_blocks = mcu_cols * component['h_sampling']
                v_blocks = mcu_rows * component['v_sampling']
                component_data[component['id']] = [[0] * (h_blocks * 8) for _ in range(v_blocks * 8)]
            
            # 重新组织解后的数据块
            mcu_index = 0
            for mcu_row in range(mcu_rows):
                for mcu_col in range(mcu_cols):
                    for comp_idx, component in enumerate(self.components):
                        block = decoded_blocks[mcu_index][comp_idx]
                        
                        # 计算块在组件中的位置
                        block_row = mcu_row * component['v_sampling']
                        block_col = mcu_col * component['h_sampling']
                        
                        # 将块数据复制到对应位置
                        for i in range(8):
                            for j in range(8):
                                row = block_row * 8 + i
                                col = block_col * 8 + j
                                if row < len(component_data[component['id']]) and \
                                   col < len(component_data[component['id']][0]):
                                    component_data[component['id']][row][col] = block[i * 8 + j]
                
                    mcu_index += 1
            
            self.decoded_data = component_data
            print("数据块重组完成")
            
            print("开始反量化处理...")
            self.dequantized_data = self._dequantize_data()
            print("反量化处理完成")
            
            print("开始IDCT变换...")
            self.idct_data = self._idct_transform()
            print("IDCT变换完成")
            
            print("开始颜色空间转换...")
            self.rgb_data = self._color_convert()
            print("颜色空间转换完成")
            
            print("开写入BMP文件...")
            self._write_bmp(bmp_path)
            print("BMP文件写入完成")
            
        except Exception as e:
            print(f"解码过程中出错：{str(e)}")
            raise

    def _decode_huffman_data(self, scan_components: List[Dict]) -> List[List[List[int]]]:
        """解码霍夫曼编码的数据"""
        decoded_data = []
        mcu_rows = (self.height + 7) // 8
        mcu_cols = (self.width + 7) // 8
        
        print(f"MCU大小: {mcu_rows}x{mcu_cols}")
        print(f"位流长度: {len(self.bits_data)}")
        
        for mcu_row in range(mcu_rows):
            for mcu_col in range(mcu_cols):
                try:
                    mcu_data = []
                    for component in scan_components:
                        block = self._decode_block(
                            component['dc_table_id'],
                            component['ac_table_id'],
                            component['id'] - 1
                        )
                        mcu_data.append(block)
                    decoded_data.append(mcu_data)
                    
                    if mcu_row == 0 and mcu_col == 0:
                        print(f"第一个MCU块的DC系数: {mcu_data[0][0]}")
                        
                except Exception as e:
                    print(f"MCU {mcu_row}x{mcu_col} 解码失败: {str(e)}")
                    mcu_data = [[0] * 64 for _ in scan_components]
                    decoded_data.append(mcu_data)
        
        return decoded_data

    def _decode_block(self, dc_table_id: int, ac_table_id: int, component_id: int) -> List[int]:
        """解码一个8x8块"""
        block = [0] * 64
        
        try:
            # 保存原始位流状态
            original_bits = self.bits_data.copy()
            original_dc_pred = self.dc_pred[component_id]
            
            # 解码DC系数
            dc_value = self._decode_dc_coefficient(dc_table_id)
            if dc_value is None:  # 检查DC解码是否成功
                self.bits_data = original_bits
                return [0] * 64
            
            self.dc_pred[component_id] += dc_value
            block[0] = self.dc_pred[component_id]
            
            # 解码AC系数
            success = self._decode_ac_coefficients(ac_table_id, block)
            if not success:  # 检查AC解码是否成功
                self.bits_data = original_bits
                self.dc_pred[component_id] = original_dc_pred
                return [0] * 64
            
            return block
            
        except Exception as e:
            print(f"块解码错误：{str(e)}, DC表ID: {dc_table_id}, AC表ID: {ac_table_id}")
            return [0] * 64

    def _decode_dc_coefficient(self, table_id: int) -> int:
        """解码DC系数"""
        try:
            dc_table = self.huffman_tables['dc'][table_id]
            code = ''
            max_length = max(len(k) for k in dc_table.keys())
            
            # 打印调试信息
            print(f"\nDC解码 - 表ID: {table_id}")
            print(f"可用的DC编码: {list(dc_table.keys())[:5]}...")
            
            # 读取霍夫曼编码
            original_bits = self.bits_data.copy()  # 保存原始位流
            bits_read = []  # 记录读取的位
            
            for _ in range(max_length):
                if len(self.bits_data) == 0:
                    print("位流数据不足")
                    self.bits_data = original_bits
                    return 0
                
                bit = int(self.bits_data[0])
                bits_read.append(bit)
                code += str(bit)
                self.bits_data = self.bits_data[1:]
                
                if code in dc_table:
                    size = dc_table[code]
                    print(f"找到匹配的霍夫曼编码: {code} -> size: {size}")
                    
                    if size == 0:
                        print("DC系数为0")
                        return 0
                    
                    # 读取差分值
                    if len(self.bits_data) < size:
                        print(f"位流数据不足以读取差分值，需要{size}位")
                        self.bits_data = original_bits
                        return 0
                    
                    # 读取size位的值
                    value = 0
                    value_bits = []
                    for _ in range(size):
                        bit = int(self.bits_data[0])
                        value_bits.append(bit)
                        value = (value << 1) | bit
                        self.bits_data = self.bits_data[1:]
                    
                    # 处理负数
                    if value < (1 << (size - 1)):
                        value = value - (1 << size) + 1
                    
                    print(f"读取到差分值: {value} (bits: {''.join(map(str, value_bits))})")
                    return value
            
            # 如果没有找到匹配的编码
            print(f"未找到匹配的霍夫曼编码，已读取的位: {''.join(map(str, bits_read))}")
            self.bits_data = original_bits
            return 0
            
        except Exception as e:
            print(f"DC解码错误：{str(e)}, 表ID: {table_id}, 当前代码: {code}")
            return 0

    def _decode_ac_coefficients(self, table_id: int, block: List[int]) -> bool:
        """解码AC系数"""
        try:
            ac_table = self.huffman_tables['ac'][table_id]
            pos = 1
            max_length = max(len(k) for k in ac_table.keys())
            
            while pos < 64:
                code = ''
                # 读取霍夫曼编码
                for _ in range(max_length):
                    if len(self.bits_data) == 0:
                        return True  # 数据结束，认为是正常的
                    
                    code += str(self.bits_data[0])
                    self.bits_data = self.bits_data[1:]
                    
                    if code in ac_table:
                        value = ac_table[code]
                        break
                else:
                    return False  # 没有找到有效的霍夫曼编码
                
                if value == 0x00:  # EOB
                    while pos < 64:
                        block[pos] = 0
                        pos += 1
                    return True
                
                run_length = (value >> 4) & 0x0F
                size = value & 0x0F
                
                pos += run_length
                if pos >= 64:
                    return False
                
                if size > 0:
                    if len(self.bits_data) < size:
                        return False
                    
                    coeff = 0
                    for _ in range(size):
                        coeff = (coeff << 1) | self.bits_data[0]
                        self.bits_data = self.bits_data[1:]
                    
                    if coeff < (1 << (size - 1)):
                        coeff = coeff - (1 << size) + 1
                    
                    block[pos] = coeff
                
                pos += 1
            
            return True
            
        except Exception as e:
            print(f"AC解码错误：{str(e)}, 表ID: {table_id}")
            return False

    def _read_signed_value(self, size: int) -> int:
        """读取有符号值"""
        try:
            if not self.bits_data.size or self.bits_data.size < size:
                raise IOError("位流数据不足")
            
            value = 0
            for _ in range(size):
                value = (value << 1) | self.bits_data[0]
                self.bits_data = self.bits_data[1:]
            
            if value < (1 << (size - 1)):
                value = value + (1 << size) - (1 << (size + 1)) + 1
            
            return value
            
        except Exception as e:
            raise IOError(f"读取有符号值时出错：{str(e)}")

    def _dequantize_data(self) -> Dict[int, List[List[float]]]:
        """反量化处理"""
        dequantized_data = {}
        
        for component in self.components:
            comp_id = component['id']
            qt_table_id = component['qt_table_id']
            quantization_table = self.quantization_tables[qt_table_id]
            
            print(f"量化表 {qt_table_id} 的前几个值: {quantization_table[:8]}")
            
            height = len(self.decoded_data[comp_id])
            width = len(self.decoded_data[comp_id][0])
            
            dequantized_data[comp_id] = [[0.0] * width for _ in range(height)]
            
            # 对每个8x8的块进行处理
            for y in range(0, height, 8):
                for x in range(0, width, 8):
                    # 从之字形顺序还原为8x8块
                    block = np.zeros((8, 8), dtype=np.float32)
                    for i in range(8):
                        for j in range(8):
                            zz_pos = zigzag_order[i * 8 + j]
                            if y + i < height and x + j < width:
                                block[i, j] = self.decoded_data[comp_id][y + i][x + j] * quantization_table[zz_pos]
                    
                    # 将反量化后的数据写回数组
                    block_height = min(8, height - y)
                    block_width = min(8, width - x)
                    for i in range(block_height):
                        for j in range(block_width):
                            dequantized_data[comp_id][y + i][x + j] = block[i, j]
        
        return dequantized_data

    def _idct_transform(self) -> Dict[int, List[List[float]]]:
        """IDCT变换"""
        idct_data = {}
        
        for component in self.components:
            comp_id = component['id']
            height = len(self.dequantized_data[comp_id])
            width = len(self.dequantized_data[comp_id][0])
            
            print(f"组件 {comp_id} 的尺寸: {width}x{height}")
            print(f"第一个块的值范围: {np.min(self.dequantized_data[comp_id][:8][:8])} ~ {np.max(self.dequantized_data[comp_id][:8][:8])}")
            
            # 预计算DCT基础矩阵（只需计算一次）
            dct_matrix = np.zeros((8, 8), dtype=np.float32)
            for i in range(8):
                scale = 1/np.sqrt(2) if i == 0 else 1.0
                for j in range(8):
                    dct_matrix[i, j] = scale * np.cos((2*j + 1) * i * np.pi / 16)
            
            # 转置矩阵只需计算一次
            dct_matrix_t = dct_matrix.T
            
            # 将输入数据转换为NumPy数组
            input_data = np.array(self.dequantized_data[comp_id], dtype=np.float32)
            output_data = np.zeros((height, width), dtype=np.float32)
            
            # 批量处理8x8块
            for y0 in range(0, height, 8):
                if y0 % 80 == 0:  # 每处理10行显示一次进度
                    print(f"IDCT处理进度: {y0}/{height} 行")
                    
                for x0 in range(0, width, 8):
                    # 提取当前8x8块
                    block_height = min(8, height - y0)
                    block_width = min(8, width - x0)
                    
                    if block_height == 8 and block_width == 8:
                        # 完整块的快速处理
                        block = input_data[y0:y0+8, x0:x0+8]
                        # 使用矩阵乘法进行IDCT
                        idct_block = np.dot(dct_matrix_t, np.dot(block, dct_matrix)) / 4.0
                    else:
                        # 处理边缘不完整的块
                        temp_block = np.zeros((8, 8), dtype=np.float32)
                        temp_block[:block_height, :block_width] = input_data[y0:y0+block_height, x0:x0+block_width]
                        idct_block = np.dot(dct_matrix_t, np.dot(temp_block, dct_matrix)) / 4.0
                        idct_block = idct_block[:block_height, :block_width]
                    
                    # 添加128限制在0-255范围内
                    idct_block = np.clip(idct_block + 128.0, 0, 255)
                    
                    # 将结果写回输出数组
                    output_data[y0:y0+block_height, x0:x0+block_width] = idct_block
            
            # 转换为整数并存储结果
            idct_data[comp_id] = output_data.astype(np.uint8).tolist()
            print(f"完成组件 {comp_id} 的IDCT变换")
        
        return idct_data

    def _color_convert(self) -> List[List[List[int]]]:
        """YCbCr转RGB颜色空间转换"""
        try:
            # 将YCbCr数据转换为NumPy数组
            y_data = np.array(self.idct_data[1], dtype=np.float32)
            cb_data = np.array(self.idct_data[2], dtype=np.float32)
            cr_data = np.array(self.idct_data[3], dtype=np.float32)
            
            print(f"Y shape: {y_data.shape}, Cb shape: {cb_data.shape}, Cr shape: {cr_data.shape}")
            print(f"Target shape: {self.height}x{self.width}")
            
            # 调整所有分量到目标尺寸
            if y_data.shape != (self.height, self.width):
                # 使用resize调整Y分量大小
                y_resized = np.zeros((self.height, self.width), dtype=np.float32)
                h_scale = self.height / y_data.shape[0]
                w_scale = self.width / y_data.shape[1]
                
                for i in range(self.height):
                    for j in range(self.width):
                        src_i = min(int(i / h_scale), y_data.shape[0] - 1)
                        src_j = min(int(j / w_scale), y_data.shape[1] - 1)
                        y_resized[i, j] = y_data[src_i, src_j]
                y_data = y_resized
            
            # 调整色度分量大小
            cb_resized = np.zeros((self.height, self.width), dtype=np.float32)
            cr_resized = np.zeros((self.height, self.width), dtype=np.float32)
            
            h_scale = self.height / cb_data.shape[0]
            w_scale = self.width / cb_data.shape[1]
            
            for i in range(self.height):
                for j in range(self.width):
                    src_i = min(int(i / h_scale), cb_data.shape[0] - 1)
                    src_j = min(int(j / w_scale), cb_data.shape[1] - 1)
                    cb_resized[i, j] = cb_data[src_i, src_j]
                    cr_resized[i, j] = cr_data[src_i, src_j]
            
            # 创建RGB图像数组
            rgb_data = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # YCbCr转RGB (使用BT.601标准)
            rgb_data[:,:,0] = np.clip(y_data + 1.402 * (cr_resized - 128), 0, 255)  # R
            rgb_data[:,:,1] = np.clip(y_data - 0.344136 * (cb_resized - 128) - 0.714136 * (cr_resized - 128), 0, 255)  # G
            rgb_data[:,:,2] = np.clip(y_data + 1.772 * (cb_resized - 128), 0, 255)  # B
            
            return rgb_data.tolist()
            
        except Exception as e:
            print(f"颜色转换错误：")
            print(f"Y形状={y_data.shape if 'y_data' in locals() else 'undefined'}")
            print(f"Cb形状={cb_data.shape if 'cb_data' in locals() else 'undefined'}")
            print(f"Cr形状={cr_data.shape if 'cr_data' in locals() else 'undefined'}")
            print(f"目标尺寸：高度={self.height}, 宽度={self.width}")
            raise Exception(f"颜色空间转换失败：{str(e)}")

    def _write_bmp(self, bmp_path: str) -> None:
        """将RGB数据保存为BMP文件"""
        header_size = 14
        dib_header_size = 40
        
        row_size = self.width * 3
        padding_size = (4 - (row_size % 4)) % 4
        padded_row_size = row_size + padding_size
        
        pixel_data_size = padded_row_size * self.height
        
        file_size = header_size + dib_header_size + pixel_data_size
        
        with open(bmp_path, 'wb') as f:
            f.write(b'BM')
            f.write(struct.pack('<I', file_size))
            f.write(struct.pack('<I', 0))
            f.write(struct.pack('<I', header_size + dib_header_size))
            
            f.write(struct.pack('<I', dib_header_size))
            f.write(struct.pack('<i', self.width))
            f.write(struct.pack('<i', -self.height))
            f.write(struct.pack('<H', 1))
            f.write(struct.pack('<H', 24))
            f.write(struct.pack('<I', 0))
            f.write(struct.pack('<I', pixel_data_size))
            f.write(struct.pack('<i', 0))
            f.write(struct.pack('<i', 0))
            f.write(struct.pack('<I', 0))
            f.write(struct.pack('<I', 0))
            
            padding = b'\x00' * padding_size
            for y in range(self.height):
                for x in range(self.width):
                    r = self.rgb_data[y][x][0]
                    g = self.rgb_data[y][x][1]
                    b = self.rgb_data[y][x][2]
                    f.write(struct.pack('<BBB', b, g, r))
                if padding_size > 0:
                    f.write(padding)

def main():
    try:
        decoder = JPEGDecoder()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        possible_extensions = ['.jpg', '.jpeg', '.JPG', '.JPEG']
        input_file = None
        
        for ext in possible_extensions:
            test_path = os.path.join(current_dir, f'input{ext}')
            if os.path.exists(test_path):
                input_file = test_path
                break
        
        if input_file is None:
            raise FileNotFoundError("找不到JPEG图片文件，请确保图片文件名为'input'且扩展名为.jpg或.jpeg")
            
        output_path = os.path.join(current_dir, 'output.bmp')
        
        print(f"正在处理文件：{input_file}")
        decoder.decode(input_file, output_path)
        print("解码完成！")
    except Exception as e:
        print(f"程序执行出错：{str(e)}")

if __name__ == '__main__':
    main()
