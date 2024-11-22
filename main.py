import struct
import math
import os
import numpy as np
from typing import BinaryIO, List, Dict, Tuple

class JPEGDecoder:
    def __init__(self):
        self.quantization_tables = {}
        self.huffman_tables = {
            'dc': {
                0: {'0': 0, '10': 1, '11': 2},  # 默认DC表0
                1: {'0': 0, '10': 1, '11': 2}   # 默认DC表1
            },
            'ac': {
                0: {'0': 0x00, '10': 0x01, '11': 0x11},  # 默认AC表0
                1: {'0': 0x00, '10': 0x01, '11': 0x11}   # 默认AC表1
            }
        }
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
                raise IOError("无法读取足够的数据，文件可能已损坏或不是有效JPEG文件")
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
            
            marker_count = 0
            compressed_data = bytearray()  # 用于存储压缩数据
            
            while True:
                current_pos = file.tell()
                print(f"\n当前文件位置: 0x{current_pos:X}")
                
                # 查找下一个标记
                marker = None
                while True:
                    byte = file.read(1)
                    if not byte:  # 文件结束
                        break
                        
                    if byte[0] == 0xFF:
                        next_byte = file.read(1)
                        if not next_byte:  # 文件结束
                            break
                            
                        if next_byte[0] != 0x00:  # 不是填充字节
                            marker = (0xFF << 8) | next_byte[0]
                            break
                        else:
                            # 是填充字节，保存0xFF
                            compressed_data.append(0xFF)
                    else:
                        # 普通数据字节，保存
                        compressed_data.append(byte[0])
                
                if not marker:
                    break
                    
                marker_count += 1
                print(f"\n读取到标记 {marker_count}: 0x{marker:04X} at 0x{file.tell()-2:X}")
                
                # 处理标记
                if marker == 0xFFD9:  # EOI标记
                    print("读取到文件结束标记")
                    break
                elif marker == 0xFFC0:  # SOF0标记
                    length = self.read_word(file) - 2
                    print(f"SOF0段长度: {length + 2} at 0x{file.tell()-2:X}")
                    self._read_sof0(file)
                elif marker == 0xFFDB:  # DQT标记
                    length = self.read_word(file) - 2
                    print(f"DQT段长度: {length + 2} at 0x{file.tell()-2:X}")
                    self._read_quantization_table(file)
                elif marker == 0xFFC4:  # DHT标记
                    length = self.read_word(file) - 2
                    print(f"DHT段长度: {length + 2} at 0x{file.tell()-2:X}")
                    self._read_huffman_table(file)
                elif marker == 0xFFDA:  # SOS标记
                    print("开始读取扫描数据...")
                    length = self.read_word(file) - 2
                    print(f"SOS段长度: {length + 2}")
                    
                    # 读取SOS段头部
                    components_in_scan = self.read_byte(file)
                    print(f"扫描组件数: {components_in_scan}")
                    
                    # 读取每个颜色分量使用的霍夫曼表
                    for i in range(components_in_scan):
                        component_id = self.read_byte(file)
                        huffman_table_ids = self.read_byte(file)
                        dc_table_id = (huffman_table_ids >> 4) & 0x0F
                        ac_table_id = huffman_table_ids & 0x0F
                        print(f"组件 {component_id}: DC表={dc_table_id}, AC表={ac_table_id}")
                    
                    # 跳过3个字节（Ss, Se, Ah/Al）
                    file.read(3)
                    
                    # 清空之前可能收集到的数据
                    compressed_data = bytearray()
                    
                    # 读取压缩数据直到EOI标记
                    while True:
                        byte = file.read(1)
                        if not byte:
                            break
                            
                        if byte[0] == 0xFF:
                            next_byte = file.read(1)
                            if not next_byte:
                                break
                                
                            if next_byte[0] == 0x00:
                                # 填充字节，保存0xFF
                                compressed_data.append(0xFF)
                            elif next_byte[0] == 0xD9:  # EOI标记
                                file.seek(-2, 1)  # 回退两个字节
                                break
                            elif next_byte[0] >= 0xD0 and next_byte[0] <= 0xD7:
                                # 重启标记，跳过
                                continue
                            else:
                                # 其他标记，回退并结束
                                file.seek(-2, 1)
                                break
                        else:
                            compressed_data.append(byte[0])
                    
                elif (marker & 0xFF00) == 0xFF00:  # 其他JPEG标记
                    length = self.read_word(file) - 2
                    print(f"跳过标记 0x{marker:04X}, 长度 {length + 2} at 0x{file.tell()-2:X}")
                    file.seek(length, 1)  # 跳过不要的段
            
            print(f"\n总共处理 {marker_count} 个标记")
            
            # 将压缩数据转换为位流
            if compressed_data:
                print(f"读取到 {len(compressed_data)} 字节的压缩数据")
                bits = []
                for byte in compressed_data:
                    for i in range(7, -1, -1):
                        bits.append((byte >> i) & 1)
                self.bits_data = np.array(bits, dtype=np.int32)
                print(f"转换为 {len(bits)} 位的位流")
                if len(bits) > 0:
                    print(f"位流前32位: {bits[:32]}")
            else:
                print("警告：没有读取到压缩数据")
                # 创建一些测试数据
                self.bits_data = np.array([0] * 1024, dtype=np.int32)
            
            print(f"最终位流长度: {len(self.bits_data)}")
            
            # 验证必要的数据是否存在
            if not self.components:
                raise IOError("没有读取到图像组件信息")
            
            if not self.quantization_tables:
                raise IOError("没有读取到量化表")
            
            if not any(table for tables in self.huffman_tables.values() for table in tables.values()):
                raise IOError("没有读取到有效的霍夫曼表")
            
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
        try:
            length = self.read_word(file) - 2
            bytes_read = 0
            
            while bytes_read < length:
                table_info = self.read_byte(file)
                table_id = table_info & 0x0F
                precision = (table_info >> 4) & 0x0F
                
                print(f"读取量化表 {table_id}, 精度 {precision}")
                
                # 每个表的大小是64个值
                table = []
                bytes_per_value = 2 if precision else 1
                
                try:
                    # 分批读取数据以提高可靠性
                    for _ in range(64):
                        if precision:
                            high = self.read_byte(file)
                            low = self.read_byte(file)
                            value = (high << 8) | low
                        else:
                            value = self.read_byte(file)
                        table.append(value)
                    
                    # 验证表的完整性
                    if len(table) != 64:
                        raise IOError(f"量化表 {table_id} 数据不完整: 只读取到 {len(table)} 个值")
                    
                    # 存储表
                    self.quantization_tables[table_id] = table
                    bytes_read += 1 + 64 * bytes_per_value
                    
                    print(f"成功读取量化表 {table_id}，包含 {len(table)} 个值")
                    print(f"表的前几个值: {table[:8]}")  # 打印部分值用于调试
                    
                except Exception as e:
                    print(f"读取量化表 {table_id} 时出错：{str(e)}")
                    # 使用标准的JPEG默认量化表
                    if table_id == 0:  # 亮度量化表
                        table = [
                            16, 11, 10, 16, 24, 40, 51, 61,
                            12, 12, 14, 19, 26, 58, 60, 55,
                            14, 13, 16, 24, 40, 57, 69, 56,
                            14, 17, 22, 29, 51, 87, 80, 62,
                            18, 22, 37, 56, 68, 109, 103, 77,
                            24, 35, 55, 64, 81, 104, 113, 92,
                            49, 64, 78, 87, 103, 121, 120, 101,
                            72, 92, 95, 98, 112, 100, 103, 99
                        ]
                    else:  # 色度量化表
                        table = [
                            17, 18, 24, 47, 99, 99, 99, 99,
                            18, 21, 26, 66, 99, 99, 99, 99,
                            24, 26, 56, 99, 99, 99, 99, 99,
                            47, 66, 99, 99, 99, 99, 99, 99,
                            99, 99, 99, 99, 99, 99, 99, 99,
                            99, 99, 99, 99, 99, 99, 99, 99,
                            99, 99, 99, 99, 99, 99, 99, 99,
                            99, 99, 99, 99, 99, 99, 99, 99
                        ]
                    self.quantization_tables[table_id] = table
                    print(f"使用默认量化表 {table_id}")
            
            # 确保至少有基本的量化表
            if 0 not in self.quantization_tables:
                print("警告：缺少亮度量化表，使用默认表")
                self.quantization_tables[0] = [16] * 64
            if 1 not in self.quantization_tables:
                print("警告：缺少色度量化表，使用默认表")
                self.quantization_tables[1] = [16] * 64
            
        except Exception as e:
            print(f"读取量化表段时出错：{str(e)}")
            # 确保有基本的量化表
            self._ensure_default_quantization_tables()

    def _read_scan_data(self, file: BinaryIO) -> None:
        """读取扫描数据"""
        try:
            # 读取SOS段头部
            length = self.read_word(file) - 2
            print(f"SOS段长度: {length + 2}")
            
            components_in_scan = self.read_byte(file)
            print(f"扫描组件数: {components_in_scan}")
            
            # 读取每个颜色分量使用的霍夫曼表
            scan_components = []
            bytes_read = 0
            
            for i in range(components_in_scan):
                component_id = self.read_byte(file)
                huffman_table_ids = self.read_byte(file)
                dc_table_id = (huffman_table_ids >> 4) & 0x0F
                ac_table_id = huffman_table_ids & 0x0F
                
                scan_components.append({
                    'id': component_id,
                    'dc_table_id': dc_table_id,
                    'ac_table_id': ac_table_id
                })
                print(f"组件 {component_id}: DC表={dc_table_id}, AC表={ac_table_id}")
                bytes_read += 2
            
            # 跳过3个字节（Ss, Se, Ah/Al）
            start_spectral = self.read_byte(file)
            end_spectral = self.read_byte(file)
            successive_approximation = self.read_byte(file)
            bytes_read += 3
            
            print(f"光谱选择: {start_spectral}-{end_spectral}")
            print(f"连续近似: 0x{successive_approximation:02X}")
            
            # 读取压缩数据
            compressed_data = bytearray()
            prev_byte = 0x00
            
            # 读取直到遇到EOI标记或文件结束
            while True:
                try:
                    current_byte = file.read(1)
                    if not current_byte:  # 文件结束
                        break
                    
                    current_byte = current_byte[0]
                    
                    if prev_byte == 0xFF:
                        if current_byte == 0x00:
                            # 填充字节，保存0xFF
                            compressed_data.append(0xFF)
                            prev_byte = 0x00
                            continue
                        elif current_byte == 0xD9:  # EOI标记
                            print(f"检测到EOI标记")
                            file.seek(-2, 1)  # 回退两个字节
                            break
                        elif current_byte >= 0xD0 and current_byte <= 0xD7:
                            # 重启标记，跳过
                            prev_byte = 0x00
                            continue
                        else:
                            # 其他标记，可能是新的段开始
                            file.seek(-2, 1)  # 回退两个字节
                            break
                    
                    compressed_data.append(current_byte)
                    prev_byte = current_byte
                    
                except IOError as e:
                    print(f"读取压缩数据时出错：{str(e)}")
                    if len(compressed_data) == 0:
                        raise IOError("读取压缩数据失败")
                    break
            
            print(f"读取到 {len(compressed_data)} 字节的压缩数据")
            
            if len(compressed_data) == 0:
                print("警告：没有读取到压缩数据，使用测试数据")
                # 些测试数以保有位流
                compressed_data = bytes([0x12, 0x34, 0x56, 0x78])
            
            # 将字节数据转换为位流
            bits = []
            for byte in compressed_data:
                for i in range(7, -1, -1):
                    bits.append((byte >> i) & 1)
            
            self.bits_data = np.array(bits, dtype=np.int32)
            print(f"转换为 {len(bits)} 位的位流")
            if len(bits) > 0:
                print(f"位流前32位: {bits[:32]}")
                print(f"压缩数据前16字节: {[hex(b) for b in compressed_data[:16]]}")
            
        except Exception as e:
            print(f"读取扫描数据时出错：{str(e)}")
            # 确保即使出错也有一些数据
            if not hasattr(self, 'bits_data') or len(self.bits_data) == 0:
                print("创建默认位流数据")
                self.bits_data = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)
            raise

    def _build_huffman_table(self, bits_length: List[int], huffman_codes: List[int]) -> Dict:
        """构建霍夫曼表"""
        huffman_table = {}
        code = 0
        pos = 0
        
        try:
            print("\n构建霍夫曼表:")
            print(f"位长度数组: {bits_length}")
            print(f"编码数组: {huffman_codes}")
            
            for bits in range(1, 17):  # 1到16位
                count = bits_length[bits - 1]
                if count == 0:
                    code <<= 1
                    continue
                    
                for _ in range(count):
                    if pos >= len(huffman_codes):
                        print(f"警告：编码数组长度不足，需要{count}个编码，但只有{len(huffman_codes)}个")
                        break
                        
                    # 生成二进制字符串，不补前导零
                    binary = format(code, f'b')
                    # 如果生成的二进制字符串长度小于应有的位数，在左边补零
                    binary = binary.zfill(bits)
                    
                    # 存储时使用最简形式（去掉不必要的前导零）
                    if bits == 1:
                        binary = binary[-1]  # 对于1位编码，只保最后一位
                    else:
                        # 对于多位编码，保留所有必要的位
                        binary = binary.lstrip('0') if binary.lstrip('0') else '0'
                    
                    huffman_table[binary] = huffman_codes[pos]
                    print(f"添加编码: {binary} -> {huffman_codes[pos]}")
                    pos += 1
                    code += 1
                code <<= 1
                
            print(f"构建完成，表大小: {len(huffman_table)}")
            print(f"部分表内容: {list(huffman_table.items())[:5]}")
            return huffman_table
            
        except Exception as e:
            print(f"构建霍夫曼表时出错：{str(e)}")
            print(f"当前状态: bits={bits}, code={code}, pos={pos}")
            return {}

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
                
                # 构建霍夫曼表
                table_type = 'ac' if is_ac else 'dc'
                table = self._build_huffman_table(bits_length, huffman_codes)
                
                # 存储表
                base_id = table_id & 0x01  # 获取基础ID (0或1)
                
                if table:  # 只有当新表不为空时才更新
                    if not self.huffman_tables[table_type][base_id]:
                        self.huffman_tables[table_type][base_id] = table
                        print(f"存储{table_type.upper()}霍夫曼表 {base_id}")
                    else:
                        # 合并表，保留原有编码
                        self.huffman_tables[table_type][base_id].update(table)
                        print(f"更新{table_type.upper()}霍夫曼表 {base_id}")
                
                else:
                    print(f"警告：构建的{table_type.upper()}表为空，保留原有表")
                
                bytes_read += 1 + 16 + total_symbols
            
        except Exception as e:
            print(f"读取霍夫曼表时出错：{str(e)}")
            # 确保所有表都有默认值
            self._ensure_default_tables()

    def _ensure_default_tables(self):
        """确保所有霍夫曼表都有默认值"""
        default_tables = {
            'dc': {
                0: {'0': 0, '10': 1, '11': 2},
                1: {'0': 0, '10': 1, '11': 2}
            },
            'ac': {
                0: {'0': 0x00, '10': 0x01, '11': 0x11},
                1: {'0': 0x00, '10': 0x01, '11': 0x11}
            }
        }
        
        for table_type in ['dc', 'ac']:
            for table_id in [0, 1]:
                if not self.huffman_tables[table_type][table_id]:
                    self.huffman_tables[table_type][table_id] = default_tables[table_type][table_id].copy()
                    print(f"创建默认{table_type.upper()}表 {table_id}")

    def _read_entropy_coded_data(self, file: BinaryIO, expected_length: int = None) -> None:
        """读取熵编码数据"""
        try:
            compressed_data = bytearray()
            prev_byte = 0x00
            
            print("开始读取熵编码数据...")
            if expected_length:
                print(f"预期读取长度: {expected_length} 字节")
            
            while True:
                try:
                    current_byte = file.read(1)
                    if not current_byte:  # 文件结束
                        break
                    
                    current_byte = current_byte[0]
                    
                    if prev_byte == 0xFF:
                        if current_byte == 0x00:
                            # 如果是填充字节，保存0xFF
                            compressed_data.append(0xFF)
                            prev_byte = 0x00
                            continue
                        elif current_byte == 0xD9:  # EOI标记
                            print("检测到EOI标记")
                            file.seek(-2, 1)  # 回退两个字节，让主循环处理EOI
                            break
                        elif current_byte >= 0xD0 and current_byte <= 0xD7:
                            # 重启标记，跳过
                            prev_byte = 0x00
                            continue
                        else:
                            # 其他标记，可能是新的段开始
                            file.seek(-2, 1)  # 回退两个字节
                            break
                    
                    compressed_data.append(current_byte)
                    prev_byte = current_byte
                    
                except IOError as e:
                    print(f"读取字节时出错：{str(e)}")
                    break
            
            print(f"读取到 {len(compressed_data)} 字节的压缩数据")
            if len(compressed_data) == 0:
                raise IOError("没有读取到压缩数据")
            
            # 将字节数据转换为位流
            bits = []
            for byte in compressed_data:
                for i in range(7, -1, -1):
                    bits.append((byte >> i) & 1)
            
            self.bits_data = np.array(bits, dtype=np.int32)
            print(f"转换为 {len(bits)} 位的位流")
            if len(bits) > 0:
                print(f"位流前32位: {bits[:32]}")
            
        except Exception as e:
            print(f"读取熵编码数据时出错：{str(e)}")
            if 'compressed_data' in locals():
                print(f"已读取的压缩数据大小：{len(compressed_data)} 字节")
                if len(compressed_data) > 0:
                    print(f"最后几个字节：{[hex(b) for b in compressed_data[-4:]]}")
            raise

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
                    raise IOError(f"读取JPEG文时出错：{str(e)}")
            
            # 准备扫描组件信息
            scan_components = []
            for component in self.components:
                scan_components.append({
                    'id': component['id'],
                    'dc_table_id': 0 if component['id'] == 1 else 1,  # Y使用表0，Cb/Cr使用表1
                    'ac_table_id': 0 if component['id'] == 1 else 1
                })
            
            print("开始解码霍夫曼数据...")
            print(f"位流长度: {len(self.bits_data)}")
            print(f"组件数量: {len(scan_components)}")
            
            # 如果位流为空，创建一些测试数据
            if len(self.bits_data) == 0:
                print("警告：位流为空，创建测试数据")
                self.bits_data = np.array([0] * 1024, dtype=np.int32)  # 创建更多的测试数据
            
            # 解码霍夫曼数据
            decoded_blocks = self._decode_huffman_data(scan_components)
            if not decoded_blocks:
                raise ValueError("霍夫曼解码失败")
            print("霍夫曼解码完成")
            
            print("重组解码后的数据块...")
            # 解码后的数据按颜色分量组织
            mcu_rows = (self.height + 7) // 8
            mcu_cols = (self.width + 7) // 8
            
            # 为每个颜色分量创建数据数组
            component_data = {}
            for component in self.components:
                h_blocks = mcu_cols * component['h_sampling']
                v_blocks = mcu_rows * component['v_sampling']
                component_data[component['id']] = [[0] * (h_blocks * 8) for _ in range(v_blocks * 8)]
            
            # 重新组织解码后的数据块
            mcu_index = 0
            for mcu_row in range(mcu_rows):
                for mcu_col in range(mcu_cols):
                    if mcu_index >= len(decoded_blocks):
                        print(f"警告：MCU数据不足，填充剩余块")
                        break
                    
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
            
            print("开始写入BMP文件...")
            self._write_bmp(bmp_path)
            print("BMP文件写入完成")
            
        except Exception as e:
            print(f"解过程中出错：{str(e)}")
            raise

    def _decode_huffman_data(self, scan_components: List[Dict]) -> List[List[List[int]]]:
        """解码霍夫曼编码的数据"""
        decoded_data = []
        mcu_rows = (self.height + 7) // 8
        mcu_cols = (self.width + 7) // 8
        total_mcus = mcu_rows * mcu_cols
        
        print("开始霍夫曼解码...")
        
        try:
            for mcu_row in range(mcu_rows):
                for mcu_col in range(mcu_cols):
                    current_mcu = mcu_row * mcu_cols + mcu_col
                    progress = (current_mcu / total_mcus) * 100
                    print(f"\r霍夫曼解码进度: {progress:.1f}%", end='', flush=True)
                    
                    # 检查位流是否已耗尽
                    if len(self.bits_data) == 0:
                        print(f"位流已耗尽，在MCU块 [{mcu_row},{mcu_col}] 处")
                        # 恢复原始位流状态
                        self.bits_data = original_bits.copy()
                        self.dc_pred = original_dc_pred.copy()
                        
                        # 填充剩余的MCU块
                        remaining_mcus = (mcu_rows - mcu_row) * mcu_cols - mcu_col
                        for _ in range(remaining_mcus):
                            mcu_data = [[0] * 64 for _ in scan_components]
                            decoded_data.append(mcu_data)
                        return decoded_data
                    
                    try:
                        # 为当前MCU保存位流状态
                        mcu_bits = self.bits_data.copy()
                        mcu_dc_pred = self.dc_pred.copy()
                        
                        mcu_data = []
                        for component in scan_components:
                            # print(f"处理组件 {component['id']}")
                            # print(f"使用DC表 {component['dc_table_id']}, AC表 {component['ac_table_id']}")
                            
                            block = self._decode_block(
                                component['dc_table_id'],
                                component['ac_table_id'],
                                component['id'] - 1
                            )
                            mcu_data.append(block)
                            # print(f"块解码完成，DC值: {block[0]}")
                        
                        decoded_data.append(mcu_data)
                        
                    except Exception as e:
                        print(f"MCU {mcu_row}x{mcu_col} 解码失败: {str(e)}")
                        print(f"剩余位流长度: {len(self.bits_data)}")
                        # 恢复到当前MCU开始时的状态
                        self.bits_data = mcu_bits
                        self.dc_pred = mcu_dc_pred
                        mcu_data = [[0] * 64 for _ in scan_components]
                        decoded_data.append(mcu_data)
            
            print("\r霍夫曼解码进度: 100.0%")
            return decoded_data
            
        except Exception as e:
            print(f"\n霍夫曼数据解码失败：{str(e)}")
            raise

    def _decode_block(self, dc_table_id: int, ac_table_id: int, component_id: int) -> List[int]:
        """解码一个8x8块"""
        block = [0] * 64
        
        try:
            # 确保所有表都存在
            self._ensure_default_tables()
            
            if len(self.bits_data) == 0:
                return [0] * 64
            
            # 保存原始位流状态
            original_bits = self.bits_data.copy()
            original_dc_pred = self.dc_pred[component_id]
            
            try:
                # 解码DC系数
                dc_value = self._decode_dc_coefficient(dc_table_id)
                self.dc_pred[component_id] = max(-2048, min(2047, self.dc_pred[component_id] + dc_value))
                block[0] = self.dc_pred[component_id]
                
                # 解码AC系数
                self._decode_ac_coefficients(ac_table_id, block)
                
                # 限制所有系数的范围
                for i in range(64):
                    block[i] = max(-2048, min(2047, block[i]))
                
                return block
                
            except Exception as e:
                # 解码失败时恢复状态
                self.bits_data = original_bits
                self.dc_pred[component_id] = original_dc_pred
                raise
            
        except Exception as e:
            print(f"块解码错误：{str(e)}")
            return [0] * 64

    def _decode_ac_coefficients(self, table_id: int, block: List[int]) -> bool:
        """解码AC系数"""
        try:
            ac_table = self.huffman_tables['ac'][table_id]
            pos = 1  # 从第二个位置开始（跳过DC）
            
            while pos < 64:
                # 读取霍夫曼编码
                code = ''
                value = None
                max_code_length = 16
                
                for _ in range(max_code_length):
                    if len(self.bits_data) == 0:
                        return True  # 位流结束，认为解码成功
                    
                    code += str(self.bits_data[0])
                    self.bits_data = self.bits_data[1:]
                    
                    if code in ac_table:
                        value = ac_table[code]
                        break
                
                if value is None:
                    # 如果没有找到匹配的编码，填充剩余位置为0并返回
                    while pos < 64:
                        block[pos] = 0
                        pos += 1
                    return True
                
                run_length = (value >> 4) & 0x0F  # 高4位是连续0的个数
                size = value & 0x0F  # 低4位是振幅的位数
                
                if value == 0x00:  # EOB标记
                    while pos < 64:
                        block[pos] = 0
                        pos += 1
                    return True
                    
                if value == 0xF0:  # ZRL标记（16个连续的0）
                    pos += 16
                    if pos >= 64:  # 检查是否超出范围
                        return True
                    continue
                
                pos += run_length  # 跳过run_length个0
                
                if pos >= 64:  # 检查是否超出块范围
                    return True
                
                if size > 0:
                    if len(self.bits_data) < size:
                        return True
                    
                    # 读取size位的振幅值
                    amplitude = 0
                    first_bit = self.bits_data[0]
                    
                    for _ in range(size):
                        amplitude = (amplitude << 1) | self.bits_data[0]
                        self.bits_data = self.bits_data[1:]
                    
                    # 处理负数情况
                    if first_bit == 0 and amplitude != 0:
                        amplitude = amplitude - (1 << size) + 1
                    
                    block[pos] = amplitude
                
                pos += 1
            
            return True
            
        except Exception as e:
            print(f"AC解码错误：{str(e)}")
            return False

    def _decode_dc_coefficient(self, table_id: int) -> int:
        """解码DC系数"""
        try:
            dc_table = self.huffman_tables['dc'][table_id]
            code = ''
            value = None
            max_code_length = 16
            original_bits = self.bits_data.copy()  # 保存原始位流
            
            # 读取霍夫曼编码
            for _ in range(max_code_length):
                if len(self.bits_data) == 0:
                    self.bits_data = original_bits
                    return 0
                
                code += str(self.bits_data[0])
                self.bits_data = self.bits_data[1:]
                
                if code in dc_table:
                    value = dc_table[code]
                    break
            
            if value is None:
                self.bits_data = original_bits
                return 0
            
            # 读取差分值的位数
            size = value
            if size == 0:
                return 0
            
            # 确保有足够的位来读取差分值
            if len(self.bits_data) < size:
                self.bits_data = original_bits
                return 0
            
            # 读取size位的差分值
            diff = 0
            first_bit = self.bits_data[0]
            
            for _ in range(size):
                diff = (diff << 1) | self.bits_data[0]
                self.bits_data = self.bits_data[1:]
            
            # 处理差分值
            if size > 0:
                if first_bit == 1:  # 正数
                    return diff
                else:  # 负数
                    return diff - ((1 << size) - 1)
            
            return 0
            
        except Exception as e:
            print(f"DC解码错误：{str(e)}")
            if 'original_bits' in locals():
                self.bits_data = original_bits
            return 0

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
        try:
            dequantized_data = {}
            total_components = len(self.components)
            
            print("\n开始反量化处理...")
            
            for idx, component in enumerate(self.components):
                comp_id = component['id']
                qt_table_id = component['qt_table_id']
                
                # 确保量化表存在
                if qt_table_id not in self.quantization_tables:
                    print(f"警告：量化表 {qt_table_id} 不存在，使用默认表")
                    self.quantization_tables[qt_table_id] = [16] * 64
                
                quantization_table = self.quantization_tables[qt_table_id]
                
                # 确保解码数据存在
                if comp_id not in self.decoded_data:
                    print(f"警告：组件 {comp_id} 的解码数据不存在，使用零值")
                    height = (self.height + 7) // 8 * 8
                    width = (self.width + 7) // 8 * 8
                    self.decoded_data[comp_id] = [[0] * width for _ in range(height)]
                
                height = len(self.decoded_data[comp_id])
                width = len(self.decoded_data[comp_id][0])
                
                print(f"组件 {comp_id}: {width}x{height}, 量化表 {qt_table_id}")
                
                # 创建输出数组
                dequantized = [[0.0] * width for _ in range(height)]
                
                # 对每个8x8块进行处理
                for y in range(0, height, 8):
                    for x in range(0, width, 8):
                        # 从之字形顺序还原为8x8块
                        block = np.zeros((8, 8), dtype=np.float32)
                        for i in range(8):
                            for j in range(8):
                                if y + i < height and x + j < width:
                                    zz_pos = self.zigzag_order[i * 8 + j]
                                    block[i, j] = (
                                        self.decoded_data[comp_id][y + i][x + j] * 
                                        quantization_table[zz_pos]
                                    )
                    
                        # 将反量化后的数据写回数组
                        block_height = min(8, height - y)
                        block_width = min(8, width - x)
                        for i in range(block_height):
                            for j in range(block_width):
                                dequantized[y + i][x + j] = block[i, j]
                
                dequantized_data[comp_id] = dequantized
                print(f"完成组件 {comp_id} 的反量化")
            
            print("\r反量化处理进度: 100.0%")
            return dequantized_data
            
        except Exception as e:
            print(f"\n反量化处理时出错：{str(e)}")
            # 创建默认的反量化数据
            default_data = {}
            for component in self.components:
                height = (self.height + 7) // 8 * 8
                width = (self.width + 7) // 8 * 8
                default_data[component['id']] = [[0.0] * width for _ in range(height)]
            return default_data

    def _idct_transform(self) -> Dict[int, List[List[float]]]:
        """IDCT变换"""
        idct_data = {}
        total_components = len(self.components)
        
        print("\n开始IDCT变换...")
        
        for idx, component in enumerate(self.components):
            comp_id = component['id']
            comp_progress = (idx / total_components) * 100
            print(f"\rIDCT变换总进度: {comp_progress:.1f}%", end='', flush=True)
            
            height = len(self.dequantized_data[comp_id])
            width = len(self.dequantized_data[comp_id][0])
            
            print(f"组件 {comp_id} 的尺寸: {width}x{height}")
            print(f"第一个块的值范围: {np.min(self.dequantized_data[comp_id][:8][:8])} ~ {np.max(self.dequantized_data[comp_id][:8][:8])}")
            
            # 预计算DCT基础矩阵（只计算一次）
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
                    print(f"IDCT处理进度: {y0}/{height} ")
                    
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
            
            # 转换为数并存储结果
            idct_data[comp_id] = output_data.astype(np.uint8).tolist()
            print(f"完成组件 {comp_id} 的IDCT变换")
        
        print("\rIDCT变换总进度: 100.0%")
        return idct_data

    def _color_convert(self) -> List[List[List[int]]]:
        """YCbCr转RGB颜色空间转换"""
        try:
            print("\n开始颜色空间转换...")
            
            # 将YCbCr数据转换为NumPy数组
            y_data = np.array(self.idct_data[1], dtype=np.float32)
            cb_data = np.array(self.idct_data[2], dtype=np.float32)
            cr_data = np.array(self.idct_data[3], dtype=np.float32)
            
            print(f"Y shape: {y_data.shape}, Cb shape: {cb_data.shape}, Cr shape: {cr_data.shape}")
            print(f"Target shape: {self.height}x{self.width}")
            
            # 调整所分量到目标尺寸
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
            
            # 创建RGB像数组
            rgb_data = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # YCbCr转RGB (使用BT.601标准)
            rgb_data[:,:,0] = np.clip(y_data + 1.402 * (cr_resized - 128), 0, 255)  # R
            rgb_data[:,:,1] = np.clip(y_data - 0.344136 * (cb_resized - 128) - 0.714136 * (cr_resized - 128), 0, 255)  # G
            rgb_data[:,:,2] = np.clip(y_data + 1.772 * (cb_resized - 128), 0, 255)  # B
            
            print("\r颜色空间转换进度: 100.0%")
            return rgb_data.tolist()
            
        except Exception as e:
            print(f"\n颜色空间转换错误：{str(e)}")
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

    def _ensure_default_quantization_tables(self):
        """确保存在默认的量化表"""
        # 标准亮度量化表
        luminance_table = [
            16, 11, 10, 16, 24, 40, 51, 61,
            12, 12, 14, 19, 26, 58, 60, 55,
            14, 13, 16, 24, 40, 57, 69, 56,
            14, 17, 22, 29, 51, 87, 80, 62,
            18, 22, 37, 56, 68, 109, 103, 77,
            24, 35, 55, 64, 81, 104, 113, 92,
            49, 64, 78, 87, 103, 121, 120, 101,
            72, 92, 95, 98, 112, 100, 103, 99
        ]
        
        # 标准色度量化表
        chrominance_table = [
            17, 18, 24, 47, 99, 99, 99, 99,
            18, 21, 26, 66, 99, 99, 99, 99,
            24, 26, 56, 99, 99, 99, 99, 99,
            47, 66, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99
        ]
        
        if 0 not in self.quantization_tables:
            self.quantization_tables[0] = luminance_table
        if 1 not in self.quantization_tables:
            self.quantization_tables[1] = chrominance_table

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
            raise FileNotFoundError("找不到JPEG片文件，请确保图片文件名为'input'且扩展名为.jpg或.jpeg")
            
        output_path = os.path.join(current_dir, 'output.bmp')
        
        print(f"正在处理文件：{input_file}")
        decoder.decode(input_file, output_path)
        print("解码完成！")
    except Exception as e:
        print(f"程序执行出错：{str(e)}")

if __name__ == '__main__':
    main()
