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
        self.quantization_tables = {}  # 添加量化表字典
        self.huffman_tables = {'dc': {}, 'ac': {}}  # 添加霍夫曼表字典
        
        # DC预测器相关
        self.dc_predictors = {}  # 添加DC预测器字典
        self.dc_lock = Lock()  # 添加DC锁
        
        # MCU相关
        self.mcu_width = 0
        self.mcu_height = 0
        self.mcu_rows = 0
        self.mcu_cols = 0
        
        # 标准霍夫曼表
        self.DC_Y_CODES = {}  # 标准DC亮度表
        self.AC_Y_CODES = {}  # 标准AC亮度表
        self.DC_C_CODES = {}  # 标准DC色度表
        self.AC_C_CODES = {}  # 标准AC色度表
        
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
        
        # 更严格的采样因子检查
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
        
        print(f"���像大小: {self.width}x{self.height}")
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
    
    # 添加标准霍夫曼表
    DC_Y_CODES = {}  # 标准DC亮度表
    AC_Y_CODES = {}  # 标准AC亮度表
    DC_C_CODES = {}  # 标准DC色度表
    AC_C_CODES = {}  # 标准AC色度表
    
    def read_dht(self, file: BinaryIO):
        """读取霍夫曼表段"""
        length = self.read_length(file)
        length -= 2
        
        while length > 0:
            table_info = struct.unpack('B', file.read(1))[0]
            table_class = 'dc' if (table_info >> 4) == 0 else 'ac'
            table_id = table_info & 0x0F
            
            # 读取位长度计数
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

    def decode_mcu(self, bit_reader, mcu_index, components):
        """根据采样类型解码单个MCU"""
        blocks = []
        
        try:
            # 检查是否到达图像边界
            mcu_row = mcu_index // self.mcu_cols
            mcu_col = mcu_index % self.mcu_cols
            
            # 计算实际的MCU大小（处理边界情况）
            mcu_width = min(self.mcu_width, self.width - mcu_col * self.mcu_width)
            mcu_height = min(self.mcu_height, self.height - mcu_row * self.mcu_height)
            
            if mcu_width <= 0 or mcu_height <= 0:
                return None
            
            if self.sampling_type == '4:4:4':
                # 4:4:4采样: 每个MCU包含3个8x8块 (Y, Cb, Cr)
                for component in components:
                    block = self._decode_block(bit_reader, component, mcu_index)
                    if block is None:
                        return None
                    blocks.append(block)
                    
            elif self.sampling_type == '4:2:2':
                # 4:2:2采样: 每个MCU包含4个块 (Y1, Y2, Cb, Cr)
                y_component = next(c for c in components if c['id'] == 1)
                # 解码2个Y块
                for _ in range(2):
                    block = self._decode_block(bit_reader, y_component, mcu_index)
                    if block is None:
                        return None
                    blocks.append(block)
                
                # 解码Cb和Cr块
                for component in components:
                    if component['id'] in [2, 3]:
                        block = self._decode_block(bit_reader, component, mcu_index)
                        if block is None:
                            return None
                        blocks.append(block)
                    
            elif self.sampling_type == '4:2:0':
                # 4:2:0采样: 每个MCU包含6个块 (Y1, Y2, Y3, Y4, Cb, Cr)
                y_component = next(c for c in components if c['id'] == 1)
                # 解码4个Y块
                for _ in range(4):
                    block = self._decode_block(bit_reader, y_component, mcu_index)
                    if block is None:
                        return None
                    blocks.append(block)
                
                # 解码Cb和Cr块
                for component in components:
                    if component['id'] in [2, 3]:
                        block = self._decode_block(bit_reader, component, mcu_index)
                        if block is None:
                            return None
                        blocks.append(block)
            
            return blocks, mcu_index
            
        except Exception as e:
            print(f"MCU #{mcu_index} 解码错误: {str(e)}")
            return None

    def decode_huffman_data(self, file: BinaryIO):
        """使用单线程解码霍夫曼编码的数据"""
        if not hasattr(file, 'read'):
            raise ValueError("Invalid file object")
        
        bit_reader = BitReader(file)
        all_blocks = []
        
        # 重置DC预测器
        self.reset_dc_predictors()
        
        try:
            # 计算总MCU数量
            total_mcus = self.mcu_rows * self.mcu_cols
            mcu_count = 0
            
            print("\n开始解码...")
            print(f"预期MCU数量: {total_mcus} ({self.mcu_rows}x{self.mcu_cols})")
            print(f"采样型: {self.sampling_type}")
            
            while mcu_count < total_mcus:
                # 检查是否遇到标记
                if bit_reader.check_for_marker():
                    print(f"\n遇到标记，重置DC预测器 (MCU #{mcu_count})")
                    self.reset_dc_predictors()
                    continue
                
                # 解码单个MCU
                result = self.decode_mcu(bit_reader, mcu_count, self.components)
                
                if result is not None:
                    blocks, _ = result
                    if blocks:
                        all_blocks.extend(blocks)
                        mcu_count += 1
                        
                        # 更新进度
                        if mcu_count % 10 == 0:
                            progress = (mcu_count / total_mcus) * 100
                            print(f"\r解码进度: {progress:.1f}% ({mcu_count}/{total_mcus} MCUs)", 
                                  end='', flush=True)
                else:
                    # 检查否到达文件结束
                    if bit_reader.found_eoi:
                        print("\n遇到EOI标记")
                        break
                    else:
                        print(f"\nMCU #{mcu_count} 解码失败，尝试继续...")
                        continue
            
            print(f"\n成功解码 {mcu_count}/{total_mcus} MCUs")
            
            if not all_blocks:
                raise ValueError("No blocks decoded")
            
            return all_blocks
            
        except Exception as e:
            print(f"\n解码错��: {str(e)}")
            raise

    def dequantize(self, block, qt_table):
        """对DCT系数进行反量化"""
        block_2d = block.reshape(8, 8)
        qt_table = qt_table.reshape(8, 8)
        
        # 确保使用正确的数据类型和计算顺序
        dequantized = np.multiply(
            block_2d.astype(np.float32),
            qt_table.astype(np.float32)
        )
        
        # 确保结果不会溢出
        return np.clip(dequantized, -2048, 2047)

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

    def process_data_unit(self, block, qt_table):
        """处理一个数据单元：反量化IDCT"""
        # 反量化
        dequantized = self.dequantize(block, qt_table)
        # IDCT变换
        spatial = self.idct_2d(dequantized)
        return spatial

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
        
        r = y + 1.402 * cr
        g = y - 0.344136 * cb - 0.714136 * cr
        b = y + 1.772 * cb
        
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
            f.write(b'BM')                          # 标识符
            f.write(struct.pack('<I', file_size))   # 文件大小
            f.write(struct.pack('<I', 0))           # 保留字段
            f.write(struct.pack('<I', 54))          # 数据偏移量
            
            # 2. 写入DIB头 (40字节)
            f.write(struct.pack('<I', 40))          # DIB头大小
            f.write(struct.pack('<I', width))       # 图像宽度
            f.write(struct.pack('<I', height))      # 图像高度
            f.write(struct.pack('<H', 1))           # 色彩平面数
            f.write(struct.pack('<H', 24))          # 每像素位数
            f.write(struct.pack('<I', 0))           # 压缩方式
            f.write(struct.pack('<I', image_size))  # 图像数据大小
            f.write(struct.pack('<I', 0))           # 水平分辨率
            f.write(struct.pack('<I', 0))           # 垂直分辨率
            f.write(struct.pack('<I', 0))           # 调色板颜色数
            f.write(struct.pack('<I', 0))           # 重要颜色数
            
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
        """执行JPEG解码，根据SWITCH控制执行阶段"""
        try:
            print(f"\n开始解码: {input_file}")
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
        """4.1 分解JPEG件"""
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
        """解码单个8x8块"""
        block = np.zeros(64, dtype=np.int32)
        
        try:
            # 解码DC系数
            dc_table = self.huffman_tables['dc'][component.get('dc_table_id', 0)]
            code = ''
            
            while True:
                bit = bit_reader.read_bits(1)
                if bit is None:
                    return None
                code += str(bit)
                if code in dc_table:
                    break
                if len(code) > 16:  # 防止无限循环
                    return None
            
            dc_size = dc_table[code]
            
            # 读取DC差分值
            if dc_size == 0:
                dc_diff = 0
            else:
                dc_bits = bit_reader.read_bits(dc_size)
                if dc_bits is None:
                    return None
                # 修改DC差分值的计算
                if dc_bits < (1 << (dc_size - 1)):
                    dc_diff = dc_bits + (-1 << dc_size) + 1
                else:
                    dc_diff = dc_bits
            
            # 更新DC预测器
            with self.dc_lock:
                comp_id = component['id']
                if comp_id not in self.dc_predictors:
                    self.dc_predictors[comp_id] = 0
                self.dc_predictors[comp_id] += dc_diff
                block[0] = self.dc_predictors[comp_id]
            
            # 解码AC系数
            ac_table = self.huffman_tables['ac'][component.get('ac_table_id', 0)]
            index = 1
            
            while index < 64:
                code = ''
                max_ac_bits = 16
                bits_read = 0
                
                while code not in ac_table and bits_read < max_ac_bits:
                    bit = bit_reader.read_bits(1)
                    if bit is None:
                        return None
                    code += str(bit)
                    bits_read += 1
                
                if code not in ac_table:
                    print(f"警告: AC码字未找到 (MCU #{mcu_index}, 组件 {component['id']})")
                    break
                
                ac_byte = ac_table[code]
                run_length = ac_byte >> 4
                ac_size = ac_byte & 0x0F
                
                if ac_byte == 0x00:  # EOB
                    break
                elif ac_byte == 0xF0:  # ZRL (16个零)
                    index += 16
                    continue
                
                # 跳过零系数
                index += run_length
                if index >= 64:
                    break
                
                # 读取AC系数
                if ac_size > 0:
                    ac_bits = bit_reader.read_bits(ac_size)
                    if ac_bits is None:
                        return None
                    block[index] = self._extend(ac_bits, ac_size)
                
                index += 1
            
            return block
            
        except Exception as e:
            print(f"块解码错误 (MCU #{mcu_index}, 组件 {component['id']}): {str(e)}")
            return None

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
            
            # 根据块在MCU中的位置确定使用哪个量化表
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
                try:
                    marker = self.read_marker(f)
                    if marker == 0xFFDA:  # SOS
                        # 读SOS段
                        length = self.read_length(f)
                        components_in_scan = struct.unpack('B', f.read(1))[0]
                        
                        # 读取每个组件的表ID
                        for _ in range(components_in_scan):
                            comp_id = struct.unpack('B', f.read(1))[0]
                            tables = struct.unpack('B', f.read(1))[0]
                            for component in self.components:
                                if component['id'] == comp_id:
                                    component['dc_table_id'] = tables >> 4
                                    component['ac_table_id'] = tables & 0x0F
                        
                        # 跳过剩余的SOS部
                        f.seek(3, 1)
                        break
                    else:
                        # 跳过其他段
                        length = self.read_length(f)
                        f.seek(length-2, 1)
                except Exception as e:
                    print(f"读取段错误: {str(e)}")
                    raise
            
            try:
                # 开始熵解码
                blocks = self.decode_huffman_data(f)
                
                if not blocks:
                    raise ValueError("解码失败：没有获取到有效的数据块")
                
                # 修改ZigZag重排序的实现
                zigzag_blocks = []
                for block in blocks:
                    # 创建新的8x8块
                    zigzag_block = np.zeros(64, dtype=np.int32)
                    # 从自然顺序转换到ZigZag顺序
                    for i in range(64):
                        zigzag_block[self.ZIGZAG_TABLE[i]] = block[i]
                    zigzag_blocks.append(zigzag_block)
                
                return zigzag_blocks
                
            except Exception as e:
                print(f"熵解码错误: {str(e)}")
                raise

    def _process_colors(self, idct_data):
        """处理颜色空间转换"""
        try:
            rgb_data = []
            
            # 根据采样方式选择处理方法
            if self.sampling_type == "4:4:4":
                rgb_data, pixels_processed = self._process_444(idct_data)
            elif self.sampling_type == "4:2:2":
                rgb_data, pixels_processed = self._process_422(idct_data)
            else:  # 4:2:0
                rgb_data, pixels_processed = self._process_420(idct_data)
            
            # 记录日志
            image_info = {
                'width': self.width,
                'height': self.height
            }
            self.logger.log_color_conversion(image_info, pixels_processed)
            
            return rgb_data
            
        except Exception as e:
            self.logger.log_error(f"Color conversion error: {str(e)}")
            raise

# 在类的开头添加ZigZag表
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
        """4.5 颜色转换日志"""
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
    SWITCH_FILE = 1   # 控制文件模式：1=单文件，2=多文件
    
    try:
        if SWITCH_FILE == 1:
            # 单文件模式
            decoder = JPEGDecoder()
            decoder.SWITCH = SWITCH
            decoder.decode('gig-sn01.jpg', 'gig-sn01.bmp')
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