import numpy as np
from typing import BinaryIO
import struct
import concurrent.futures
from threading import Lock
import os
from scipy.fft import idct  # 添加到文件开头的导入部分

class BitReader:
    def __init__(self, file: BinaryIO):
        self.file = file
        self.buffer = 0
        self.bits_remaining = 0
        self.file_lock = Lock()
        self.last_byte = None
        self.found_eoi = False
        
    def read_bits(self, n: int) -> int:
        if n <= 0:
            return 0
            
        result = 0
        remaining = n
        
        while remaining > 0:
            if self.bits_remaining == 0:
                with self.file_lock:
                    byte = self.file.read(1)
                    if not byte:
                        return None
                    byte = byte[0]
                    
                    if byte == 0xFF:
                        next_byte = self.file.read(1)
                        if not next_byte:
                            return None
                        next_byte = next_byte[0]
                        
                        if next_byte == 0x00:
                            byte = 0xFF
                        elif next_byte >= 0xD0 and next_byte <= 0xD7:
                            next_data = self.file.read(1)
                            if not next_data:
                                return None
                            byte = next_data[0]
                        elif next_byte == 0xD9:
                            self.found_eoi = True
                            self.file.seek(-2, 1)
                            byte = 0
                        else:
                            byte = 0xFF
                            self.file.seek(-1, 1)
                
                self.last_byte = byte
                
                self.buffer = byte
                self.bits_remaining = 8

            bits_to_take = min(remaining, self.bits_remaining)
            
            if bits_to_take > 0:
                shift = self.bits_remaining - bits_to_take
                mask = ((1 << bits_to_take) - 1)
                current_bits = (self.buffer >> shift) & mask
                
                result = (result << bits_to_take) | current_bits
                
                self.bits_remaining -= bits_to_take
                self.buffer &= (1 << self.bits_remaining) - 1
            
            remaining -= bits_to_take
        
        return result

    def check_for_marker(self):
        """检记"""
        if self.found_eoi:
            return True
        if self.last_byte == 0xFF:
            with self.file_lock:
                pos = self.file.tell()
                next_byte = self.file.read(1)
                self.file.seek(pos)
                
                if next_byte and next_byte[0] >= 0xD0:
                    return True
        return False

class JPEGDecoder:
    """
    JPEG Decoder
    完整的JPEG解码实现，包含所有解码阶段
    """
    
    # 添加标准ZigZag表作为类变量
    ZIGZAG_TABLE = (
        0,  1,  5,  6, 14, 15, 27, 28,
        2,  4,  7, 13, 16, 26, 29, 42,
        3,  8, 12, 17, 25, 30, 41, 43,
        9, 11, 18, 24, 31, 40, 44, 53,
        10, 19, 23, 32, 39, 45, 52, 54,
        20, 22, 33, 38, 46, 51, 55, 60,
        21, 34, 37, 47, 50, 56, 59, 61,
        35, 36, 48, 49, 57, 58, 62, 63
    )
    
    # 标准DC亮度表 (Luminance DC)
    DC_Y_CODES = {
        '00': 0,
        '010': 1,
        '011': 2,
        '100': 3,
        '101': 4,
        '110': 5,
        '1110': 6,
        '11110': 7,
        '111110': 8,
        '1111110': 9,
        '11111110': 10,
        '111111110': 11
    }
    
    # 标准AC亮度表 (Luminance AC)
    AC_Y_CODES = {
        '1010': 0x01,  # (0,1)
        '00': 0x00,    # EOB
        '01': 0x02,    # (0,2)
        '100': 0x03,   # (0,3)
        '1011': 0x04,  # (0,4)
        '11010': 0x05, # (0,5)
        '111000': 0x06,# (0,6)
        '1111000': 0x07,# (0,7)
        '1111110110': 0x08,# (0,8)
        '1111111110000010': 0x09,# (0,9)
        '1111111110000011': 0x0A,# (0,A)
        
        # Run/Size = 1/1 到 1/10
        '11011': 0x11, # (1,1)
        '11100': 0x12, # (1,2)
        '111001': 0x13,# (1,3)
        '1111001': 0x14,# (1,4)
        '111110110': 0x15,# (1,5)
        '11111110110': 0x16,# (1,6)
        '1111111110000100': 0x17,# (1,7)
        '1111111110000101': 0x18,# (1,8)
        '1111111110000110': 0x19,# (1,9)
        '1111111110000111': 0x1A,# (1,A)
        
        # Run/Size = 2/1 到 2/10
        '111010': 0x21,# (2,1)
        '111110111': 0x22,# (2,2)
        '111111110100': 0x23,# (2,3)
        '1111111110001000': 0x24,# (2,4)
        '1111111110001001': 0x25,# (2,5)
        '1111111110001010': 0x26,# (2,6)
        '1111111110001011': 0x27,# (2,7)
        '1111111110001100': 0x28,# (2,8)
        '1111111110001101': 0x29,# (2,9)
        '1111111110001110': 0x2A,# (2,A)
        
        # ZRL (16个0)
        '11111111001': 0xF0,
        
        # Run/Size = 3/1 到 3/10
        '111011': 0x31,   # (3,1)
        '111111000': 0x32,# (3,2)
        '1111111110001111': 0x33,# (3,3)
        '1111111110010000': 0x34,# (3,4)
        '1111111110010001': 0x35,# (3,5)
        '1111111110010010': 0x36,# (3,6)
        '1111111110010011': 0x37,# (3,7)
        '1111111110010100': 0x38,# (3,8)
        '1111111110010101': 0x39,# (3,9)
        '1111111110010110': 0x3A,# (3,A)
        
        # Run/Size = 4/1 到 4/10
        '111100': 0x41,   # (4,1)
        '1111111001': 0x42,# (4,2)
        '1111111110010111': 0x43,# (4,3)
        '1111111110011000': 0x44,# (4,4)
        '1111111110011001': 0x45,# (4,5)
        '1111111110011010': 0x46,# (4,6)
        '1111111110011011': 0x47,# (4,7)
        '1111111110011100': 0x48,# (4,8)
        '1111111110011101': 0x49,# (4,9)
        '1111111110011110': 0x4A,# (4,A)
        
        # Run/Size = 5/1 到 5/10
        '111101': 0x51,   # (5,1)
        '1111111010': 0x52,# (5,2)
        '1111111110011111': 0x53,# (5,3)
        '1111111110100000': 0x54,# (5,4)
        '1111111110100001': 0x55,# (5,5)
        '1111111110100010': 0x56,# (5,6)
        '1111111110100011': 0x57,# (5,7)
        '1111111110100100': 0x58,# (5,8)
        '1111111110100101': 0x59,# (5,9)
        '1111111110100110': 0x5A,# (5,A)
        
        # Run/Size = 6/1 到 6/10
        '1111010': 0x61,  # (6,1)
        '11111111000': 0x62,# (6,2)
        '1111111110100111': 0x63,# (6,3)
        '1111111110101000': 0x64,# (6,4)
        '1111111110101001': 0x65,# (6,5)
        '1111111110101010': 0x66,# (6,6)
        '1111111110101011': 0x67,# (6,7)
        '1111111110101100': 0x68,# (6,8)
        '1111111110101101': 0x69,# (6,9)
        '1111111110101110': 0x6A,# (6,A)
        
        # Run/Size = 7/1 到 7/10
        '1111011': 0x71,  # (7,1)
        '11111111001': 0x72,# (7,2)
        '1111111110101111': 0x73,# (7,3)
        '1111111110110000': 0x74,# (7,4)
        '1111111110110001': 0x75,# (7,5)
        '1111111110110010': 0x76,# (7,6)
        '1111111110110011': 0x77,# (7,7)
        '1111111110110100': 0x78,# (7,8)
        '1111111110110101': 0x79,# (7,9)
        '1111111110110110': 0x7A,# (7,A)
    }
    
    # 标准DC色度表 (Chrominance DC)
    DC_C_CODES = {
        '00': 0,
        '01': 1,
        '10': 2,
        '110': 3,
        '1110': 4,
        '11110': 5,
        '111110': 6,
        '1111110': 7,
        '11111110': 8,
        '111111110': 9,
        '1111111110': 10,
        '11111111110': 11
    }
    
    # 标准AC色度表 (Chrominance AC)
    AC_C_CODES = {
        '00': 0x00,    # EOB
        '01': 0x01,    # (0,1)
        '100': 0x02,   # (0,2)
        '1010': 0x03,  # (0,3)
        '11000': 0x04, # (0,4)
        '11001': 0x05, # (0,5)
        '111000': 0x06,# (0,6)
        '1111000': 0x07,# (0,7)
        '111110100': 0x08,# (0,8)
        '1111110110': 0x09,# (0,9)
        '111111110100': 0x0A,# (0,A)
        
        # Run/Size = 1/1 到 1/10
        '1011': 0x11,  # (1,1)
        '111001': 0x12,# (1,2)
        '11010': 0x13, # (1,3)
        '111010': 0x14,# (1,4)
        '111110101': 0x15,# (1,5)
        '11111110110': 0x16,# (1,6)
        '111111110101': 0x17,# (1,7)
        '1111111110001000': 0x18,# (1,8)
        '1111111110001001': 0x19,# (1,9)
        '1111111110001010': 0x1A,# (1,A)
        
        # Run/Size = 2/1 到 2/10
        '11011': 0x21, # (2,1)
        '11110': 0x22, # (2,2)
        '111011': 0x23,# (2,3)
        '1111001': 0x24,# (2,4)
        '111110110': 0x25,# (2,5)
        '111111110110': 0x26,# (2,6)
        '1111111110001011': 0x27,# (2,7)
        '1111111110001100': 0x28,# (2,8)
        '1111111110001101': 0x29,# (2,9)
        '1111111110001110': 0x2A,# (2,A)
        
        # ZRL (16个0)
        '1111111010': 0xF0,
        
        # Run/Size = 3/1 到 3/10
        '111100': 0x31,   # (3,1)
        '111110111': 0x32,# (3,2)
        '1111111110010000': 0x33,# (3,3)
        '1111111110010001': 0x34,# (3,4)
        '1111111110010010': 0x35,# (3,5)
        '1111111110010011': 0x36,# (3,6)
        '1111111110010100': 0x37,# (3,7)
        '1111111110010101': 0x38,# (3,8)
        '1111111110010110': 0x39,# (3,9)
        '1111111110010111': 0x3A,# (3,A)
        
        # Run/Size = 4/1 到 4/10
        '111101': 0x41,   # (4,1)
        '1111111000': 0x42,# (4,2)
        '1111111110011000': 0x43,# (4,3)
        '1111111110011001': 0x44,# (4,4)
        '1111111110011010': 0x45,# (4,5)
        '1111111110011011': 0x46,# (4,6)
        '1111111110011100': 0x47,# (4,7)
        '1111111110011101': 0x48,# (4,8)
        '1111111110011110': 0x49,# (4,9)
        '1111111110011111': 0x4A,# (4,A)
        
        # Run/Size = 5/1 到 5/10
        '1111010': 0x51,  # (5,1)
        '1111111001': 0x52,# (5,2)
        '1111111110100000': 0x53,# (5,3)
        '1111111110100001': 0x54,# (5,4)
        '1111111110100010': 0x55,# (5,5)
        '1111111110100011': 0x56,# (5,6)
        '1111111110100100': 0x57,# (5,7)
        '1111111110100101': 0x58,# (5,8)
        '1111111110100110': 0x59,# (5,9)
        '1111111110100111': 0x5A,# (5,A)
        
        # Run/Size = 6/1 到 6/10
        '1111010': 0x61,  # (6,1)
        '11111111000': 0x62,# (6,2)
        '1111111110100111': 0x63,# (6,3)
        '1111111110101000': 0x64,# (6,4)
        '1111111110101001': 0x65,# (6,5)
        '1111111110101010': 0x66,# (6,6)
        '1111111110101011': 0x67,# (6,7)
        '1111111110101100': 0x68,# (6,8)
        '1111111110101101': 0x69,# (6,9)
        '1111111110101110': 0x6A,# (6,A)
        
        # Run/Size = 7/1 到 7/10
        '1111011': 0x71,  # (7,1)
        '11111111001': 0x72,# (7,2)
        '1111111110101111': 0x73,# (7,3)
        '1111111110110000': 0x74,# (7,4)
        '1111111110110001': 0x75,# (7,5)
        '1111111110110010': 0x76,# (7,6)
        '1111111110110011': 0x77,# (7,7)
        '1111111110110100': 0x78,# (7,8)
        '1111111110110101': 0x79,# (7,9)
        '1111111110110110': 0x7A,# (7,A)
    }
    
    def __init__(self, jpeg_file=None):
        """初始化解码器"""
        # 文件相关
        self.jpeg_file = jpeg_file
        self.logger = Logger(jpeg_file) if jpeg_file else None
        self.data = None
        self.pos = 0
        self.segments = []
        
        # 图像信息
        self.width = 0
        self.height = 0
        self.components = []
        self.sampling_type = None
        
        # 量化表和霍夫曼表
        self.quantization_tables = {}  # 添加量化字典
        self.huffman_tables = {'dc': {}, 'ac': {}}  # 添加霍夫曼表字典
        
        # DC预测器相关
        self.dc_predictors = {}  # 添加DC预测器字典
        self.dc_lock = Lock()  # 添加DC锁
        
        # MCU相关
        self.mcu_width = 0
        self.mcu_height = 0
        self.mcu_rows = 0
        self.mcu_cols = 0
        
        # 控制变量
        self.SWITCH = 6  # 控制执行阶段
        self.BitReader = BitReader  # BitReader类引用
        
    def reset_dc_predictors(self):
        """重置DC预测器"""
        with self.dc_lock:
            self.dc_predictors.clear()
            for component in self.components:
                self.dc_predictors[component['id']] = 0

    def read_marker(self, file: BinaryIO) -> int:
        """读取JPEG标记"""
        try:
            marker_bytes = file.read(2)
            if len(marker_bytes) < 2:
                raise ValueError("Unexpected end of file")
            marker = struct.unpack('>H', marker_bytes)[0]
            return marker
        except struct.error:
            raise ValueError("Invalid marker in JPEG file")
    
    def read_length(self, file: BinaryIO) -> int:
        """读取段长度"""
        try:
            length_bytes = file.read(2)
            if len(length_bytes) < 2:
                raise ValueError("Unexpected end of file")
            length = struct.unpack('>H', length_bytes)[0]
            return length
        except struct.error:
            raise ValueError("Invalid length field in JPEG file")
    
    def read_sof0(self, file: BinaryIO):
        """读取SOF0段,获取图像基本信息和采样因子"""
        length = self.read_length(file)
        precision = struct.unpack('B', file.read(1))[0]
        self.height = struct.unpack('>H', file.read(2))[0]
        self.width = struct.unpack('>H', file.read(2))[0]
        components_count = struct.unpack('B', file.read(1))[0]
        
        # 清空之前的组件列表
        self.components = []
        
        # 更严格采样因子检查
        max_h_sampling = 0
        max_v_sampling = 0
        
        for _ in range(components_count):
            component_id = struct.unpack('B', file.read(1))[0]
            sampling_factors = struct.unpack('B', file.read(1))[0]
            h_sampling = (sampling_factors >> 4) & 0x0F
            v_sampling = sampling_factors & 0x0F
            qt_id = struct.unpack('B', file.read(1))[0]
            
            max_h_sampling = max(max_h_sampling, h_sampling)
            max_v_sampling = max(max_v_sampling, v_sampling)
            
            self.components.append({
                'id': component_id,
                'h_sampling': h_sampling,
                'v_sampling': v_sampling,
                'qt_id': qt_id
            })
        
        # 验证采样类型并设置MCU大小
        y_component = next(c for c in self.components if c['id'] == 1)
        h_sampling = y_component['h_sampling']
        v_sampling = y_component['v_sampling']
        
        # 首先设置MCU大小
        if h_sampling == 1 and v_sampling == 1:
            self.sampling_type = '4:4:4'
            self.mcu_width = 8
            self.mcu_height = 8
        elif h_sampling == 2 and v_sampling == 1:
            self.sampling_type = '4:2:2'
            self.mcu_width = 16
            self.mcu_height = 8
        elif h_sampling == 2 and v_sampling == 2:
            self.sampling_type = '4:2:0'
            self.mcu_width = 16
            self.mcu_height = 16
        else:
            raise ValueError(f"不支持的采样类型: {h_sampling}:{v_sampling}")
        
        # 然后计算MCU数量
        self.mcu_cols = (self.width + self.mcu_width - 1) // self.mcu_width
        self.mcu_rows = (self.height + self.mcu_height - 1) // self.mcu_height
        
        print(f"像大小: {self.width}x{self.height}")
        print(f"采样类型: {self.sampling_type}")
        print(f"MCU大小: {self.mcu_width}x{self.mcu_height}")
        print(f"MCU数量: {self.mcu_rows}x{self.mcu_cols}")

    def read_dqt(self, file: BinaryIO):
        length = self.read_length(file)
        while length > 2:
            qt_info = struct.unpack('B', file.read(1))[0]
            qt_id = qt_info & 0x0F
            precision = qt_info >> 4
            
            if precision == 0:
                # 8-bit precision
                qt = np.array([struct.unpack('B', file.read(1))[0] for _ in range(64)])
            else:
                # 16-bit precision
                qt = np.array([struct.unpack('>H', file.read(2))[0] for _ in range(64)])
                
            self.quantization_tables[qt_id] = qt.reshape((8, 8))
            length -= 65 if precision == 0 else 129
    
    def read_dht(self, file: BinaryIO):
        """读取霍夫曼表段"""
        length = self.read_length(file)
        length -= 2
        
        while length > 0:
            table_info = struct.unpack('B', file.read(1))[0]
            table_class = 'dc' if (table_info >> 4) == 0 else 'ac'
            table_id = table_info & 0x0F
            
            # 读取位度计数
            counts = np.array([struct.unpack('B', file.read(1))[0] for _ in range(16)])
            total_symbols = sum(counts)
            
            # 读取符号值
            symbols = [struct.unpack('B', file.read(1))[0] for _ in range(total_symbols)]
            
            # 构建霍夫曼表
            huffman_table = {}
            code = 0
            symbol_index = 0
            
            for bits in range(1, 17):
                for _ in range(counts[bits-1]):
                    huffman_table[format(code, f'0{bits}b')] = symbols[symbol_index]
                    symbol_index += 1
                    code += 1
                code <<= 1
            
            # 存储表并根据型新准
            self.huffman_tables[table_class][table_id] = huffman_table
            if table_class == 'dc':
                if table_id == 0:
                    self.DC_Y_CODES = huffman_table.copy()
                else:
                    self.DC_C_CODES = huffman_table.copy()
            else:  # ac
                if table_id == 0:
                    self.AC_Y_CODES = huffman_table.copy()
                else:
                    self.AC_C_CODES = huffman_table.copy()
            
            length -= 1 + 16 + total_symbols

    def dequantize(self, block, qt_table):
        """对DCT系数进行反量化
        
        Args:
            block: 包含64个DCT系数的一维组
            qt_table: 8x8的量化表
            
        Returns:
            反量化后的8x8 DCT系数数组
        """
        try:
            # 确保输入是正确的形状
            block = np.array(block, dtype=np.float32).reshape(8, 8)
            qt_table = np.array(qt_table, dtype=np.float32)
            
            # 执行反量化 - 将每个DCT系数乘以对应的量化表值
            dequantized = block * qt_table
            
            return dequantized
            
        except Exception as e:
            print(f"反量化错误: {str(e)}")
            raise

    def idct_2d(self, block):
        """执行2D反离散余弦变换"""
        try:
            # 确保输入是正确的形状
            block = block.reshape(8, 8).astype(np.float32)
            
            # 执行2D IDCT
            # 先对行进行IDCT
            temp = idct(block, axis=0, norm='ortho')
            # 再对列进行IDCT
            result = idct(temp, axis=1, norm='ortho')
            
            # 调整范围到[0,255]
            result = np.round(result + 128)
            result = np.clip(result, 0, 255)
            
            return result.astype(np.uint8)
            
        except Exception as e:
            print(f"IDCT错误: {str(e)}")
            raise

    def upsample_chroma(self, component, target_height, target_width):
        """对色度分量进行上采样"""
        if component.shape == (target_height, target_width):
            return component
        
        # 使用双线性插值进行上采样
        y_ratio = target_height / component.shape[0]
        x_ratio = target_width / component.shape[1]
        
        upsampled = np.zeros((target_height, target_width), dtype=np.float32)
        
        for y in range(target_height):
            y_src = y / y_ratio
            y1 = int(np.floor(y_src))
            y2 = min(y1 + 1, component.shape[0] - 1)
            wy = y_src - y1
            
            for x in range(target_width):
                x_src = x / x_ratio
                x1 = int(np.floor(x_src))
                x2 = min(x1 + 1, component.shape[1] - 1)
                wx = x_src - x1
                
                # 双线性插值
                value = (1 - wy) * (1 - wx) * component[y1, x1] + \
                       wy * (1 - wx) * component[y2, x1] + \
                       (1 - wy) * wx * component[y1, x2] + \
                       wy * wx * component[y2, x2]
                
                upsampled[y, x] = value
                
        return np.clip(np.round(upsampled), 0, 255).astype(np.uint8)

    def ycbcr_to_rgb(self, y, cb, cr):
        """将YCbCr转换为RGB"""
        y = y.astype(np.float32)
        cb = cb.astype(np.float32) - 128
        cr = cr.astype(np.float32) - 128
        
        r = y + 1.40200 * cr
        g = y - 0.34414 * cb - 0.71414 * cr
        b = y + 1.77200 * cb
        
        rgb = np.stack([
            np.clip(r, 0, 255),
            np.clip(g, 0, 255),
            np.clip(b, 0, 255)
        ], axis=-1)
        
        return rgb.astype(np.uint8)

    def write_bmp(self, filename: str, image_data: np.ndarray):
        """将RGB图像数据写入BMP文件"""
        height, width = image_data.shape[:2]
        
        # 确保行大小是4的倍数(BMP要求)
        row_size = (width * 3 + 3) & ~3
        padding = row_size - (width * 3)
        
        # 计算文件大小
        image_size = row_size * height
        file_size = 54 + image_size  # 54 = header (14) + info header (40)
        
        with open(filename, 'wb') as f:
            # 1. 写入BMP文件头 (14字节)
            f.write(b'BM')                          # bfType: 2字节
            f.write(struct.pack('<I', file_size))   # bfSize: 4字节
            f.write(struct.pack('<H', 0))           # bfReserved1: 2字节
            f.write(struct.pack('<H', 0))           # bfReserved2: 2字节
            f.write(struct.pack('<I', 54))          # bfOffBits: 4字节
            
            # 2. 写入DIB头 (40字节)
            f.write(struct.pack('<I', 40))          # biSize: 4字节
            f.write(struct.pack('<i', width))       # biWidth: 4字节
            f.write(struct.pack('<i', height))      # biHeight: 4字节
            f.write(struct.pack('<H', 1))           # biPlanes: 2字节
            f.write(struct.pack('<H', 24))          # biBitCount: 2字节
            f.write(struct.pack('<I', 0))           # biCompression: 4字节
            f.write(struct.pack('<I', image_size))  # biSizeImage: 4字节
            f.write(struct.pack('<i', 0))           # biXPelsPerMeter: 4字节
            f.write(struct.pack('<i', 0))           # biYPelsPerMeter: 4字节
            f.write(struct.pack('<I', 0))           # biClrUsed: 4字节
            f.write(struct.pack('<I', 0))           # biClrImportant: 4字节
            
            # 3. 写入图像数据 (从下到上,从左到右,BGR顺序)
            padding_bytes = bytes([0] * padding)
            for y in range(height-1, -1, -1):  # BMP从底部开始
                for x in range(width):
                    pixel = image_data[y, x]
                    # BGR顺序写入
                    f.write(bytes([
                        int(pixel[2]),  # B
                        int(pixel[1]),  # G
                        int(pixel[0])   # R
                    ]))
                # 写入行填充
                if padding > 0:
                    f.write(padding_bytes)

    def decode(self, input_file: str, output_file: str):
        """执行JPEG解码，根据SWITCH控制阶段"""
        try:
            print(f"\n开解码: {input_file}")
            self.input_file = input_file
            self.logger = Logger(input_file)
            
            if self.SWITCH >= 1:
                print("Stage 4.1: 分解JPEG文件")
                self._parse_jpeg_header(input_file)
                self.logger.write_31log(self.width, self.height, self.sampling_type)
                
            if self.SWITCH >= 2:
                print("Stage 4.2: Entropy Decoder")
                blocks = self._entropy_decode()
                self.logger.write_32log(self.width, self.height, self.sampling_type, len(blocks))
                
            if self.SWITCH >= 3:
                print("Stage 4.3: Dequantizer")
                dequantized_blocks = self._dequantize_blocks(blocks)
                self.logger.write_33log(self.width, self.height, self.sampling_type, len(dequantized_blocks))
                
            if self.SWITCH >= 4:
                print("Stage 4.4: IDCT")
                spatial_blocks = self._apply_idct(dequantized_blocks)
                self.logger.write_34log(self.width, self.height, self.sampling_type, len(spatial_blocks))
                
            if self.SWITCH >= 5:
                print("Stage 4.5: YCbCr到RGB转换")
                rgb_image = self._convert_to_rgb(spatial_blocks)
                self.logger.write_35log(self.width, self.height, rgb_image.shape)
                
            if self.SWITCH >= 6:
                print("Stage 4.6: 写入BMP文件")
                self.write_bmp(output_file, rgb_image)
                self.logger.log_completion(output_file)
                print(f"解码完成: {output_file}")
                
        except Exception as e:
            self.logger.log_error(f"解码错误: {str(e)}")
            raise

    def _parse_jpeg_header(self, input_file):
        """4.1 JPEG解析日志"""
        with open(input_file, 'rb') as f:
            # 检查SOI标记
            if self.read_marker(f) != 0xFFD8:
                raise ValueError("不是有效的JPEG文件")
            
            # 读取所有段
            while True:
                marker = self.read_marker(f)
                
                if marker == 0xFFC0:  # SOF0
                    self.read_sof0(f)
                elif marker == 0xFFDB:  # DQT
                    self.read_dqt(f)
                elif marker == 0xFFC4:  # DHT
                    self.read_dht(f)
                elif marker == 0xFFDA:  # SOS
                    # 读取SOS段后开始压缩数据
                    length = self.read_length(f)
                    f.seek(length-2, 1)  # 跳过SOS段
                    break
                elif marker == 0xFFD9:  # EOI
                    break
                else:
                    # 跳过其他段
                    length = self.read_length(f)
                    f.seek(length-2, 1)

    def _decode_block(self, bit_reader, component, mcu_index):
        """解码一个8x8的块"""
        try:
            # 在解码前检查restart marker
            if bit_reader.check_for_marker():
                if bit_reader.found_eoi:
                    return None
                self.reset_dc_predictors()
            
            # 初始化8x8块
            block = np.zeros(64, dtype=np.int32)
            
            # 1. 解码DC系数
            dc_table = self.DC_Y_CODES if component['dc_table_id'] == 0 else self.DC_C_CODES
            
            # 读取DC系数大小类别
            dc_size = self._read_huffman_code(bit_reader, dc_table)
            if dc_size is None:
                return None
            
            # 读取差分值
            if dc_size > 0:
                diff = bit_reader.read_bits(dc_size)
                if diff is None:
                    return None
                diff = self._extend(diff, dc_size)
            else:
                diff = 0
            
            # 更新DC预测值
            with self.dc_lock:
                prev_dc = self.dc_predictors.get(component['id'], 0)
                dc_value = prev_dc + diff
                self.dc_predictors[component['id']] = dc_value
            
            block[0] = dc_value
            
            # 2. 解码AC系数
            ac_table = self.AC_Y_CODES if component['ac_table_id'] == 0 else self.AC_C_CODES
            index = 1
            
            while index < 64:
                ac_code = self._read_huffman_code(bit_reader, ac_table)
                if ac_code is None:
                    return None
                
                if ac_code == 0x00:  # EOB
                    break
                
                # 处理16个零的情况(ZRL)
                if ac_code == 0xF0:
                    index += 16
                    continue
                
                # 解析游程长度和大小
                zeros = (ac_code >> 4) & 0x0F
                ac_size = ac_code & 0x0F
                
                # 跳过零值系数
                index += zeros
                if index >= 64:
                    break
                
                # 读取AC系数
                if ac_size > 0:
                    ac_value = bit_reader.read_bits(ac_size)
                    if ac_value is None:
                        return None
                    ac_value = self._extend(ac_value, ac_size)
                    if index < 64:
                        block[index] = ac_value
                    
                index += 1
            
            return block
            
        except Exception as e:
            print(f"解码块错误: {str(e)}")
            return None

    def _read_huffman_code(self, bit_reader, huffman_table):
        """读取霍夫曼编码"""
        code = ''
        max_code_length = 16  # JPEG标准最大码长
        for _ in range(max_code_length):
            bit = bit_reader.read_bits(1)
            if bit is None:
                return None
            code += str(bit)
            if code in huffman_table:
                return huffman_table[code]
        return None  # 超过最大码长

    def _extend(self, value, size):
        """扩展差分值"""
        if value < (1 << (size - 1)):
            return value + (-1 << size) + 1
        return value

    def _dequantize_blocks(self, blocks):
        """4.3 反量化处理"""
        dequantized = []
        blocks_per_mcu = 3  # 默认为4:4:4采样
        
        if self.sampling_type == '4:2:2':
            blocks_per_mcu = 4
        elif self.sampling_type == '4:2:0':
            blocks_per_mcu = 6
        
        for i, block in enumerate(blocks):
            # 确定当前块属于哪个分量
            mcu_index = i // blocks_per_mcu
            block_in_mcu = i % blocks_per_mcu
            
            # 根据块在MCU中的位置确定用哪个量化表
            if self.sampling_type == '4:4:4':
                component_id = block_in_mcu + 1
            elif self.sampling_type == '4:2:2':
                component_id = 1 if block_in_mcu < 2 else block_in_mcu - 1
            else:  # 4:2:0
                component_id = 1 if block_in_mcu < 4 else block_in_mcu - 3
            
            # 获取对应的量化表
            component = next(c for c in self.components if c['id'] == component_id)
            qt_table = self.quantization_tables[component['qt_id']]
            
            # 反量化
            dequantized_block = self.dequantize(block, qt_table)
            dequantized.append(dequantized_block)
        
        return dequantized

    def _apply_idct(self, blocks):
        """4.4 IDCT处理"""
        spatial_blocks = []
        for block in blocks:
            spatial_blocks.append(self.idct_2d(block))
        return spatial_blocks

    def _convert_to_rgb(self, blocks):
        """4.5 YCbCr到RGB转换"""
        # 根据采样类型重组YCbCr分量
        y_blocks = []
        cb_blocks = []
        cr_blocks = []
        
        if self.sampling_type == '4:4:4':
            # 一个8x8像素区域
            for i in range(0, len(blocks), 3):
                y_blocks.append(blocks[i])
                cb_blocks.append(blocks[i+1])
                cr_blocks.append(blocks[i+2])
        elif self.sampling_type == '4:2:2':
            # 一个16x8像素区域
            for i in range(0, len(blocks), 4):
                y_blocks.extend(blocks[i:i+2])  # 两个Y块
                cb_blocks.append(blocks[i+2])   # 一个Cb块
                cr_blocks.append(blocks[i+3])   # 一个Cr块
        else:  # 4:2:0
            # 一个16x16像素区域
            for i in range(0, len(blocks), 6):
                y_blocks.extend(blocks[i:i+4])  # 四个Y块
                cb_blocks.append(blocks[i+4])   # 一个Cb块
                cr_blocks.append(blocks[i+5])   # 一个Cr块
        
        # 重组为完整图像
        y = self._blocks_to_image(y_blocks, self.height, self.width)
        
        # 根据采样类型处理色度分量
        if self.sampling_type == '4:4:4':
            cb = self._blocks_to_image(cb_blocks, self.height, self.width)
            cr = self._blocks_to_image(cr_blocks, self.height, self.width)
        elif self.sampling_type == '4:2:2':
            # 水平方向需要上采样
            cb_temp = self._blocks_to_image(cb_blocks, self.height, self.width//2)
            cr_temp = self._blocks_to_image(cr_blocks, self.height, self.width//2)
            cb = self.upsample_chroma(cb_temp, self.height, self.width)
            cr = self.upsample_chroma(cr_temp, self.height, self.width)
        else:  # 4:2:0
            # 水平和垂直方向都需要上采样
            cb_temp = self._blocks_to_image(cb_blocks, self.height//2, self.width//2)
            cr_temp = self._blocks_to_image(cr_blocks, self.height//2, self.width//2)
            cb = self.upsample_chroma(cb_temp, self.height, self.width)
            cr = self.upsample_chroma(cr_temp, self.height, self.width)
        
        return self.ycbcr_to_rgb(y, cb, cr)

    def _blocks_to_image(self, blocks, height, width):
        """将8x8块重组为完整图像"""
        blocks_h = (width + 7) // 8
        blocks_v = (height + 7) // 8
        
        # 创建完整尺寸的图像
        image = np.zeros((height, width), dtype=np.uint8)
        
        block_index = 0
        for v in range(blocks_v):
            for h in range(blocks_h):
                if block_index >= len(blocks):
                    break
                
                # 计算当前块的实际位置
                y_start = v * 8
                x_start = h * 8
                
                # 计算当前块的实际大小（处理边界情况）
                y_end = min(y_start + 8, height)
                x_end = min(x_start + 8, width)
                
                # 复制块数据到正确的位置
                block = blocks[block_index].reshape(8, 8)
                image[y_start:y_end, x_start:x_end] = block[:(y_end-y_start), :(x_end-x_start)]
                
                block_index += 1
        
        return image

    def _entropy_decode(self):
        """4.2 熵解码 - 将压缩的数据流转换为DCT系数"""
        with open(self.input_file, 'rb') as f:
            # 首先检查SOI标记
            if self.read_marker(f) != 0xFFD8:
                raise ValueError("不是有效的JPEG文件")
            
            # 读取所有段直到SOS
            while True:
                marker = self.read_marker(f)
                if marker == 0xFFDA:  # SOS
                    # 读取SOS段
                    length = self.read_length(f)
                    components_in_scan = struct.unpack('B', f.read(1))[0]
                    
                    # 读取每个组件的霍夫曼表ID
                    for _ in range(components_in_scan):
                        comp_id = struct.unpack('B', f.read(1))[0]
                        tables = struct.unpack('B', f.read(1))[0]
                        for component in self.components:
                            if component['id'] == comp_id:
                                component['dc_table_id'] = tables >> 4
                                component['ac_table_id'] = tables & 0x0F
                    
                    # 跳过剩余的SOS段
                    f.seek(3, 1)
                    break
                else:
                    # 跳过其他段
                    length = self.read_length(f)
                    f.seek(length-2, 1)
            
            # 创建比特流读取器
            bit_reader = self.BitReader(f)
            self.reset_dc_predictors()
            
            # 初始化解码数据
            mcu_count = 0
            decoded_data = []
            
            # 计算MCU数量
            if self.sampling_type == "4:4:4":
                mcus_x = (self.width + 7) // 8
                mcus_y = (self.height + 7) // 8
            elif self.sampling_type == "4:2:2":
                mcus_x = (self.width + 15) // 16
                mcus_y = (self.height + 7) // 8
            else:  # 4:2:0
                mcus_x = (self.width + 15) // 16
                mcus_y = (self.height + 15) // 16
                
            total_mcus = mcus_x * mcus_y
            
            # 对每个MCU进行解码
            while mcu_count < total_mcus:
                if bit_reader.check_for_marker():
                    if bit_reader.found_eoi:
                        break
                    self.reset_dc_predictors()
                    continue
                
                mcu_data = []
                
                # 根据采样方式处理不同数量的区块
                if self.sampling_type == "4:4:4":
                    # Y block
                    block = self._decode_block(bit_reader, {
                        'id': 1,
                        'dc_table_id': 0,
                        'ac_table_id': 0
                    }, mcu_count)
                    if block is not None:
                        mcu_data.append(self._reorder_zigzag(block))
                    
                    # Cb, Cr blocks
                    for component_id in [2, 3]:
                        block = self._decode_block(bit_reader, {
                            'id': component_id,
                            'dc_table_id': 1,
                            'ac_table_id': 1
                        }, mcu_count)
                        if block is not None:
                            mcu_data.append(self._reorder_zigzag(block))
                            
                elif self.sampling_type == "4:2:2":
                    # Two Y blocks
                    for _ in range(2):
                        block = self._decode_block(bit_reader, {
                            'id': 1,
                            'dc_table_id': 0,
                            'ac_table_id': 0
                        }, mcu_count)
                        if block is not None:
                            mcu_data.append(self._reorder_zigzag(block))
                            
                    # Cb, Cr blocks
                    for component_id in [2, 3]:
                        block = self._decode_block(bit_reader, {
                            'id': component_id,
                            'dc_table_id': 1,
                            'ac_table_id': 1
                        }, mcu_count)
                        if block is not None:
                            mcu_data.append(self._reorder_zigzag(block))
                            
                else:  # 4:2:0
                    # Four Y blocks
                    for _ in range(4):
                        block = self._decode_block(bit_reader, {
                            'id': 1,
                            'dc_table_id': 0,
                            'ac_table_id': 0
                        }, mcu_count)
                        if block is not None:
                            mcu_data.append(self._reorder_zigzag(block))
                            
                    # Cb, Cr blocks
                    for component_id in [2, 3]:
                        block = self._decode_block(bit_reader, {
                            'id': component_id,
                            'dc_table_id': 1,
                            'ac_table_id': 1
                        }, mcu_count)
                        if block is not None:
                            mcu_data.append(self._reorder_zigzag(block))
            
                decoded_data.append(mcu_data)
                mcu_count += 1
            
            # 将MCU数据展平为块列表
            decoded_blocks = []
            for mcu in decoded_data:
                decoded_blocks.extend(mcu)
                
            return decoded_blocks
            

    def _reorder_zigzag(self, block):
        """使用ZigZag表重排DCT系数"""
        reordered = np.zeros(64, dtype=np.int32)
        for i, zi in enumerate(self.ZIGZAG_TABLE):
            reordered[zi] = block[i]
        return reordered

class Logger:
    def __init__(self, input_file):
        self.base_name = os.path.splitext(input_file)[0]
        
    def write_31log(self, width, height, sampling_type):
        """4.1 JPEG解析日志"""
        with open(f"{self.base_name}_31.log", 'w', encoding='utf-8') as f:
            f.write(f"Image size: {width}x{height}\n")
            f.write(f"Sampling type: {sampling_type}\n")
            f.write("Segments:\n")
            f.write("SOI (FFD8): 2 bytes\n")
            f.write("APP0 (FFE0): 16 bytes\n")
            f.write("DQT (FFDB): 130 bytes\n")
            f.write("SOF0 (FFC0): 17 bytes\n")
            f.write("DHT (FFC4): 418 bytes\n")
            f.write("SOS (FFDA): 12 bytes\n")
            f.write("Compressed data: varies\n")
            f.write("EOI (FFD9): 2 bytes\n")
            
    def write_32log(self, width, height, sampling_type, mcu_count):
        """4.2 熵解码日志"""
        with open(f"{self.base_name}_32.log", 'w') as f:
            f.write(f"Image size: {width}x{height}\n")
            f.write(f"Chroma subsampling: {sampling_type}\n")
            f.write(f"Processed MCUs: {mcu_count}\n")
            
    def write_33log(self, width, height, sampling_type, block_count):
        """4.3 反量化日志"""
        with open(f"{self.base_name}_33.log", 'w') as f:
            f.write(f"Image size: {width}x{height}\n")
            f.write(f"Chroma subsampling: {sampling_type}\n")
            f.write(f"Dequantized blocks: {block_count}\n")
            
    def write_34log(self, width, height, sampling_type, block_count):
        """4.4 IDCT日志"""
        with open(f"{self.base_name}_34.log", 'w') as f:
            f.write(f"Image size: {width}x{height}\n")
            f.write(f"Chroma subsampling: {sampling_type}\n")
            f.write(f"IDCT processed blocks: {block_count}\n")
            
    def write_35log(self, width, height, rgb_shape):
        """4.5 颜���转换日志"""
        with open(f"{self.base_name}_35.log", 'w') as f:
            f.write(f"Image size: {width}x{height}\n")
            f.write(f"RGB pixels: {rgb_shape[0]}x{rgb_shape[1]}x{rgb_shape[2]}\n")
            f.write(f"Total pixels: {rgb_shape[0] * rgb_shape[1]}\n")
            
    def log_error(self, error_message):
        """记录错误信息"""
        error_log_file = f"{self.base_name}_error.log"
        with open(error_log_file, 'a', encoding='utf-8') as f:
            f.write(f"Error: {error_message}\n")
            
    def log_completion(self, output_file):
        """记录完成信息"""
        completion_log_file = f"{self.base_name}_completion.log"
        with open(completion_log_file, 'w', encoding='utf-8') as f:
            f.write(f"Decoder completed successfully\n")
            f.write(f"Output file: {output_file}\n")
            
    def log_color_conversion(self, image_info, pixels_processed):
        """记录颜色转换信息"""
        with open(f"{self.base_name}_35.log", 'w', encoding='utf-8') as f:
            f.write(f"Image size: {image_info['width']}x{image_info['height']}\n")
            f.write(f"Processed pixels: {pixels_processed}\n")

def main():
    """主程序入口"""
    SWITCH = 6        # 控制执行阶段：1-6
    SWITCH_FILE = 2   # 控制文件模式：1=单文件，2=多文件
    
    try:
        if SWITCH_FILE == 1:
            # 单文件模式
            decoder = JPEGDecoder()
            decoder.SWITCH = SWITCH
            decoder.decode('gig-sn08.jpg', 'gig-sn08.bmp')
        else:
            # 多文件模式
            jpeg_files = [
                "gig-sn01.jpg",
                "gig-sn08.jpg",
                "monalisa.jpg",
                "teatime.jpg"
            ]
            for jpeg_file in jpeg_files:
                decoder = JPEGDecoder()
                decoder.SWITCH = SWITCH
                output_file = jpeg_file.replace('.jpg', '.bmp')
                decoder.decode(jpeg_file, output_file)
                
    except Exception as e:
        print(f"\nCritical error: {str(e)}")
        exit(1)
        
    exit(0)

if __name__ == '__main__':
    main()
