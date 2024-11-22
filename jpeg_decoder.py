from scipy.fft import idct

class Logger:
    def __init__(self):
        self.logs = {
            '31': [],  # 区段解析日志
            '32': [],  # 熵解码日志
            '33': [],  # 反量化日志
            '34': [],  # IDCT日志
            '35': []   # 颜色转换日志
        }
        
    def log(self, log_type, message):
        """记录日志"""
        self.logs[log_type].append(message)
        
    def write_log(self, log_type):
        """写入日志文件"""
        filename = f"{log_type}log"
        with open(filename, 'w') as f:
            for message in self.logs[log_type]:
                f.write(f"{message}\n")

class JPEGDecoder:
    def __init__(self, filename):
        self.filename = filename
        self.data = None
        self.pos = 0
        
        # 图像基本信息
        self.width = 0
        self.height = 0
        self.components = 0  # 颜色分量数
        self.sampling_factors = {}  # 采样因子
        self.sampling_mode = None   # 添加采样模式属性
        
        # 量化表和哈夫曼表
        self.quant_tables = {}
        self.huffman_tables = {
            'dc': {},  # DC哈夫曼表
            'ac': {}   # AC哈夫曼表
        }
        
        # MCU相关
        self.mcu_width = 0
        self.mcu_height = 0
        self.mcu_count = 0
        
        # 开关控制
        self.switch = 6  # 控制执行步骤
        
        # 创建logger实例
        self.logger = Logger()
        
    def read_file(self):
        """读取JPEG文件"""
        try:
            # 检查文件名是否为空
            if not self.filename:
                raise ValueError("文件名不能为空")
            
            # 尝试获取文件的绝对路径
            import os
            if not os.path.isabs(self.filename):
                # 如果是相对路径，转换为绝对路径
                current_dir = os.path.dirname(os.path.abspath(__file__))
                self.filename = os.path.join(current_dir, self.filename)
            
            # 检查文件是否存在
            if not os.path.exists(self.filename):
                raise FileNotFoundError(f"找不到文件: {self.filename}")
            
            # 检查文件是否可读
            if not os.access(self.filename, os.R_OK):
                raise PermissionError(f"无法读取文件: {self.filename}")
            
            # 检查文件大小
            file_size = os.path.getsize(self.filename)
            if file_size == 0:
                raise ValueError(f"文件 {self.filename} 为空")
            if file_size < 2:
                raise ValueError(f"文件 {self.filename} 太小，不是有效的JPEG文件")
            
            # 读取文件
            with open(self.filename, 'rb') as f:
                self.data = f.read()
            
            # 检查是否成功读取数据
            if not self.data:
                raise ValueError(f"无法从文件 {self.filename} 读取数据")
            
            # 检查文件头
            if len(self.data) < 2 or self.data[0] != 0xFF or self.data[1] != 0xD8:
                raise ValueError(f"文件 {self.filename} 不是有效的JPEG文件（缺少SOI标记）")
            
            # 记录日志
            self.logger.log('31', f"成功读取文件: {self.filename}")
            self.logger.log('31', f"文件大小: {len(self.data)} 字节")
            
        except FileNotFoundError:
            print(f"错误：找不到文件 {self.filename}")
            print("请确保文件存在且文件名正确")
            raise
        except PermissionError:
            print(f"错误：无法访问文件 {self.filename}")
            print("请确保有足够的文件访问权限")
            raise
        except ValueError as e:
            print(f"错误：{str(e)}")
            print("请确保提供有效的JPEG文件")
            raise
        except Exception as e:
            print(f"读取文件时发生未知错误：{str(e)}")
            print(f"文件路径：{self.filename}")
            raise
        
    def read_word(self):
        """读取2字节"""
        try:
            # 检查是否有足够的数据可读
            if self.pos + 1 >= len(self.data):
                self.logger.log('31', f"尝试读取2字节时遇到文件末尾，当前位置：{self.pos}")
                raise EOFError("已到达文件末尾")
            
            word = (self.data[self.pos] << 8) | self.data[self.pos + 1]
            self.pos += 2
            return word
            
        except IndexError:
            self.logger.log('31', f"文件数据不足，无法读取2字节，当前位置：{self.pos}")
            raise EOFError(f"文件数据不足，无法读取2字节，当前位置：{self.pos}")
        except Exception as e:
            self.logger.log('31', f"读取2字节时出错: {str(e)}")
            raise
        
    def read_byte(self):
        """读取1字节"""
        try:
            # 检查是否有足够的数据可读
            if self.pos >= len(self.data):
                self.logger.log('31', f"尝试读取1字节时遇到文件末尾，当前位置：{self.pos}")
                raise EOFError("已到达文件末尾")
            
            byte = self.data[self.pos]
            self.pos += 1
            return byte
            
        except IndexError:
            self.logger.log('31', f"文件数据不足，无法读取1字节，当前位置：{self.pos}")
            raise EOFError(f"文件数据不足，无法读取1字节，当前位置：{self.pos}")
        except Exception as e:
            self.logger.log('31', f"读取1字节时出错: {str(e)}")
            raise
        
    def parse_markers(self):
        """解析JPEG标记"""
        if self.switch < 1:
            return
            
        try:
            # 验证SOI标记
            if self.read_word() != 0xFFD8:
                raise ValueError("不是有效的JPEG文件")
            
            while self.pos < len(self.data) - 1:  # 确保至少还有2个字节可读
                try:
                    # 查找下一个标记
                    while self.pos < len(self.data):
                        byte = self.data[self.pos]
                        if byte == 0xFF and self.pos + 1 < len(self.data):
                            marker_byte = self.data[self.pos + 1]
                            if marker_byte != 0x00:  # 不是填充字节
                                break
                        self.pos += 1
                    
                    if self.pos >= len(self.data) - 1:
                        break
                    
                    # 读取完整标记
                    marker = (self.data[self.pos] << 8) | self.data[self.pos + 1]
                    self.pos += 2
                    
                    if marker == 0xFFD9:  # EOI
                        break
                    
                    # 对于除了EOI之外的标记，都应该有长度字段
                    if self.pos + 1 >= len(self.data):
                        self.logger.log('31', "文件在长度字段处意外结束")
                        break
                    
                    length = (self.data[self.pos] << 8) | self.data[self.pos + 1]
                    self.pos += 2
                    length -= 2  # 减去长度字段本身的2字节
                    
                    # 检查剩余数据是否足够
                    if self.pos + length > len(self.data):
                        self.logger.log('31', f"标记0x{marker:04X}的数据不完整")
                        break
                    
                    # 处理不同类型的标记
                    # 注意处理顺序：先处理DQT，再处理SOF0
                    if marker == 0xFFDB:  # DQT
                        self._parse_dqt()
                    elif marker == 0xFFC4:  # DHT
                        self._parse_dht()
                    elif marker == 0xFFC0:  # SOF0
                        if not self.quant_tables:
                            raise ValueError("在SOF0标记之前未找到量化表")
                        self._parse_sof0()
                    elif marker == 0xFFDA:  # SOS
                        self._parse_sos()
                        break  # SOS之后是压缩数据，需要特殊处理
                    else:
                        # 跳过其他标记的数据
                        self.pos += length
                    
                except EOFError as e:
                    self.logger.log('31', f"解析标记时遇到文件结尾: {str(e)}")
                    break
                except Exception as e:
                    self.logger.log('31', f"解析标记时出错: {str(e)}")
                    raise
                
        except Exception as e:
            self.logger.log('31', f"解析JPEG标记时出错: {str(e)}")
            raise
        
    def decode(self):
        """主解码流程，按照业务需求执行解码步骤"""
        try:
            # 4.1 分解输入的JPEG file
            print("\n步骤4.1: 分解JPEG文件...")
            self.read_file()
            self.parse_markers()
            # 生成31log
            self.logger.write_log('31')
            
            # 4.2 Entropy Decoder，输出为DCT系数
            print("\n步骤4.2: 熵解码...")
            self.entropy_decode()
            # 生成32log
            header = f"图像尺寸: {self.width}x{self.height}, 采样方式: {self.sampling_mode}"
            self.logger.log('32', header)
            self.logger.log('32', f"处理MCU数量: {self.mcu_count}")
            self.logger.write_log('32')
            
            # 4.3 Dequantizer，还原DCT系数
            print("\n步骤4.3: 反量化...")
            self.dequantize()
            # 生成33log
            header = f"图像尺寸: {self.width}x{self.height}, 采样方式: {self.sampling_mode}"
            self.logger.log('33', header)
            self.logger.log('33', f"处理MCU数量: {self.mcu_count}")
            self.logger.write_log('33')
            
            # 4.4 IDCT，将频率域转换回空间域
            print("\n步骤4.4: IDCT变换...")
            self.inverse_dct()
            # 生成34log
            header = f"图像尺寸: {self.width}x{self.height}, 采样方式: {self.sampling_mode}"
            self.logger.log('34', header)
            self.logger.log('34', f"处理MCU数量: {self.mcu_count}")
            self.logger.write_log('34')
            
            # 4.5 将YCbCr转换回RGB
            print("\n步骤4.5: 颜色空间转换...")
            self.color_convert()
            # 生成35log
            total_pixels = self.width * self.height
            self.logger.log('35', f"还原RGB像素数量: {total_pixels}")
            self.logger.log('35', f"原图像素数量: {total_pixels}")
            self.logger.write_log('35')
            
            # 4.6 写入BMP文件
            print("\n步骤4.6: 生成BMP文件...")
            self.write_bmp()
            # 输出完成信息
            output_filename = self.filename.rsplit('.', 1)[0] + '.bmp'
            print(f"Decoder completed: {output_filename}")
            
        except Exception as e:
            self.logger.log('31', f"解码过程出错: {str(e)}")
            raise

    def _parse_sof0(self):
        """解析SOF0(Start Of Frame)标记，获取图像基本信息"""
        try:
            length = self.read_word()  # 读取长度
            precision = self.read_byte()  # 精度(通常是8位)
            
            # 读取图像高度和宽度
            self.height = self.read_word()
            self.width = self.read_word()
            
            # 验证图像尺寸
            if self.width <= 0 or self.height <= 0:
                raise ValueError(f"无效的图像尺寸: {self.width}x{self.height}")
            
            # 读取颜色分量数量
            self.components = self.read_byte()  # 通常是3(YCbCr)或1(灰度图)
            if self.components not in [1, 3]:
                raise ValueError(f"不支持的颜色分量数: {self.components}")
            
            # 记录日志
            self.logger.log('31', f"图像尺寸: {self.width}x{self.height}")
            self.logger.log('31', f"颜色分量数: {self.components}")
            self.logger.log('31', f"精度: {precision}位")
            
            # 存储每个分量的信息
            max_h_factor = 0  # 最大水平采样因子
            max_v_factor = 0  # 最大垂直采样因子
            
            for i in range(self.components):
                # 读取分量ID(1=Y, 2=Cb, 3=Cr)
                component_id = self.read_byte()
                if component_id not in [1, 2, 3]:
                    raise ValueError(f"无效的分量ID: {component_id}")
                
                # 读取采样因子(高4位是垂直采样因子，低4位是水平采样因子)
                sampling = self.read_byte()
                h_factor = (sampling >> 4) & 0x0F  # 水平采样因子
                v_factor = sampling & 0x0F         # 垂直采样因子
                
                # 验证采样因子
                if h_factor == 0 or v_factor == 0:
                    raise ValueError(f"无效的采样因子: {h_factor}x{v_factor}")
                
                # 更新最大采样因子
                max_h_factor = max(max_h_factor, h_factor)
                max_v_factor = max(max_v_factor, v_factor)
                
                # 读取量化表ID
                quant_table_id = self.read_byte()
                if quant_table_id not in self.quant_tables:
                    raise ValueError(f"未找到量化表ID: {quant_table_id}")
                
                # 存储分量信息
                self.sampling_factors[component_id] = {
                    'h_factor': h_factor,
                    'v_factor': v_factor,
                    'quant_table_id': quant_table_id
                }
                
                # 记录日志
                component_name = {1: 'Y', 2: 'Cb', 3: 'Cr'}.get(component_id, f'Unknown({component_id})')
                self.logger.log('31', f"分量{component_name}: 采样因子={h_factor}x{v_factor}, 量化表ID={quant_table_id}")
            
            # 确定采样方式(4:4:4, 4:2:2, 或 4:2:0)
            if self.components == 3:  # 彩色图像
                y_sampling = self.sampling_factors[1]
                if y_sampling['h_factor'] == 1 and y_sampling['v_factor'] == 1:
                    self.sampling_mode = '4:4:4'
                elif y_sampling['h_factor'] == 2 and y_sampling['v_factor'] == 1:
                    self.sampling_mode = '4:2:2'
                elif y_sampling['h_factor'] == 2 and y_sampling['v_factor'] == 2:
                    self.sampling_mode = '4:2:0'
                else:
                    raise ValueError(f"不支持的采样方式: {y_sampling['h_factor']}x{y_sampling['v_factor']}")
                
                self.logger.log('31', f"采样方式: {self.sampling_mode}")
            
            # 计算MCU的尺寸
            self.mcu_width = max_h_factor * 8   # MCU宽度 = 最大水平采样因子 * 8
            self.mcu_height = max_v_factor * 8  # MCU高度 = 最大垂直采样因子 * 8
            
            # 计算图像包含的MCU数量
            mcu_cols = (self.width + self.mcu_width - 1) // self.mcu_width
            mcu_rows = (self.height + self.mcu_height - 1) // self.mcu_height
            self.mcu_count = mcu_cols * mcu_rows
            
            self.logger.log('31', f"MCU尺寸: {self.mcu_width}x{self.mcu_height}")
            self.logger.log('31', f"MCU数量: {self.mcu_count} ({mcu_cols}x{mcu_rows})")
            
        except Exception as e:
            self.logger.log('31', f"解析SOF0标记时出错: {str(e)}")
            raise ValueError(f"解析SOF0标记时出错: {str(e)}")
        
    def _parse_dht(self):
        """解析DHT(Define Huffman Table)标记，获取哈夫曼表信息"""
        length = self.read_word() - 2  # 去长度字段本身的2字节
        end_pos = self.pos + length
        
        while self.pos < end_pos:
            # 读取表信息字节
            table_info = self.read_byte()
            table_class = (table_info >> 4) & 0x0F  # 0=DC表, 1=AC表
            table_id = table_info & 0x0F            # 表ID (0-3)
            
            # 确定表类型
            table_type = 'dc' if table_class == 0 else 'ac'
            
            # 读取码长计数（BITS）：16个字节，表示每个码长的符号数量
            code_lengths = [self.read_byte() for _ in range(16)]
            total_symbols = sum(code_lengths)
            
            # 读取符号值（HUFFVAL）
            symbols = [self.read_byte() for _ in range(total_symbols)]
            
            # 构建哈夫曼表
            huffman_table = {}
            code = 0
            pos = 0
            
            # 为每个码长生成哈夫曼码
            for length in range(1, 17):
                num_codes = code_lengths[length - 1]
                for _ in range(num_codes):
                    if pos >= len(symbols):
                        break
                    # 生成码字的二进制字符串表示，左侧补0到指定长度
                    binary = format(code, f'0{length}b')
                    huffman_table[binary] = symbols[pos]
                    pos += 1
                    code += 1
                code <<= 1
            
            # 存储构建好的曼表
            self.huffman_tables[table_type][table_id] = huffman_table
            
            # 记录日志
            table_desc = "DC" if table_class == 0 else "AC"
            self.logger.log('31', f"解析{table_desc}哈夫曼表 ID={table_id}")
            self.logger.log('31', f"码长分布: {code_lengths}")
            self.logger.log('31', f"符号总数: {total_symbols}")

    def _parse_dqt(self):
        """解析DQT(Define Quantization Table)标记，获取量化表信息"""
        length = self.read_word() - 2  # 减去长度字段本身的2字节
        end_pos = self.pos + length
        
        # 标准JPEG zigzag顺序表
        zigzag_order = [
            0,  1,  5,  6, 14, 15, 27, 28,
            2,  4,  7, 13, 16, 26, 29, 42,
            3,  8, 12, 17, 25, 30, 41, 43,
            9, 11, 18, 24, 31, 40, 44, 53,
            10, 19, 23, 32, 39, 45, 52, 54,
            20, 22, 33, 38, 46, 51, 55, 60,
            21, 34, 37, 47, 50, 56, 59, 61,
            35, 36, 48, 49, 57, 58, 62, 63
        ]
        
        while self.pos < end_pos:
            # 读取表信息字节
            table_info = self.read_byte()
            table_id = table_info & 0x0F            # 表ID (0-3)
            precision = (table_info >> 4) & 0x0F    # 0=8位, 1=16位
            
            # 创建8x8的量化表
            quant_table = [[0] * 8 for _ in range(8)]
            
            # 读取量化值
            if precision == 0:  # 8位精度
                raw_values = [self.read_byte() for _ in range(64)]
            else:  # 16位精度
                raw_values = [self.read_word() for _ in range(64)]
                
            # 按zigzag顺序填充量化表
            for i in range(64):
                row = zigzag_order[i] // 8
                col = zigzag_order[i] % 8
                quant_table[row][col] = raw_values[i]
                
            # 存储量化表
            self.quant_tables[table_id] = quant_table
            
            # 记录日志
            self.logger.log('31', f"解析量化表 ID={table_id}")
            self.logger.log('31', f"精度: {8 if precision == 0 else 16}位")
            
            # 记录量化表的一些统计信息
            min_value = min(raw_values)
            max_value = max(raw_values)
            avg_value = sum(raw_values) / 64
            self.logger.log('31', f"量化值范围: {min_value}-{max_value}, 平均值: {avg_value:.2f}")
            
            # 打印量化表的第一行作为示例
            self.logger.log('31', f"量化表第一行: {quant_table[0]}")
        
    def _dequantize_block(self, block, quant_table):
        """对单个8x8块进行反量化"""
        result = [[0] * 8 for _ in range(8)]
        for i in range(8):
            for j in range(8):
                result[i][j] = block[i][j] * quant_table[i][j]
        return result
        
    def _get_huffman_code(self, data, start_pos, table):
        """从比特流中读取哈夫曼码"""
        code = ""
        pos = start_pos
        
        while True:
            if pos >= len(data) * 8:
                raise ValueError("数据流结束但未找到有效的哈夫曼码")
                
            # 从字节流中读取一个位
            byte_pos = pos // 8
            bit_pos = 7 - (pos % 8)  # 从高位到低位读取
            bit = (data[byte_pos] >> bit_pos) & 1
            code += str(bit)
            
            # 检查当前编码是否在哈夫曼表中
            if code in table:
                return table[code], pos + len(code)
            
            pos += 1 
        
    def _parse_sos(self):
        """解析SOS(Start Of Scan)标记，获取扫描参数"""
        length = self.read_word() - 2  # 减去长度字段本身的2字节
        
        # 取颜色分量数
        num_components = self.read_byte()
        self.logger.log('31', f"扫描分量数: {num_components}")
        
        # 存储每个分量的哈夫曼表信息
        self.scan_components = {}
        
        # 读取每个分量的信息
        for _ in range(num_components):
            # 读取分量ID
            component_id = self.read_byte()
            
            # 读取哈夫曼表选择字节
            huffman_tables = self.read_byte()
            dc_table_id = (huffman_tables >> 4) & 0x0F  # DC表ID
            ac_table_id = huffman_tables & 0x0F         # AC表ID
            
            # 存储分量信息
            self.scan_components[component_id] = {
                'dc_table_id': dc_table_id,
                'ac_table_id': ac_table_id,
                'prev_dc': 0  # 用于DC差分编码
            }
            
            # 记录日志
            component_name = {1: 'Y', 2: 'Cb', 3: 'Cr'}.get(component_id, f'Unknown({component_id})')
            self.logger.log('31', f"分量{component_name}: DC表={dc_table_id}, AC表={ac_table_id}")
        
        # 读取光谱选择参数
        start_spectral = self.read_byte()  # 通常为0
        end_spectral = self.read_byte()    # 通常为63
        approx = self.read_byte()          # 渐进式JPEG使用，baseline为0
        
        # 记录日志
        self.logger.log('31', f"光谱范围: {start_spectral}-{end_spectral}")
        
        # 寻找图像数据的起始位置
        # 跳过标记段，直找到实际的图像数据
        while True:
            byte = self.read_byte()
            if byte == 0xFF:
                next_byte = self.read_byte()
                if next_byte == 0x00:  # 如果是FF00，表示数据中的FF字节
                    self.pos -= 2  # 回退到FF的位置
                    break
                elif next_byte >= 0xD0 and next_byte <= 0xD7:  # 重启标记
                    continue
                else:
                    raise ValueError(f"意外的标记: FF{next_byte:02X}")
            else:
                self.pos -= 1  # 回退一个字节
                break
        
        # 记录图像数据的起始位置
        self.scan_data_pos = self.pos
        
        def _read_bits(self, num_bits):
            """从数据流中读取指定数量的位"""
            result = 0
            for _ in range(num_bits):
                if self.bit_pos == -1:
                    byte = self.read_byte()
                    if byte == 0xFF:
                        next_byte = self.read_byte()
                        if next_byte != 0x00:
                            raise ValueError("意外的标记")
                    self.bit_buffer = byte
                    self.bit_pos = 7
                
                result = (result << 1) | ((self.bit_buffer >> self.bit_pos) & 1)
                self.bit_pos -= 1
                
            return result
        
        def _extend_value(self, value, num_bits):
            """扩展值的符号"""
            if value < (1 << (num_bits - 1)):
                value = value + (-1 << num_bits) + 1
            return value

    def entropy_decode(self):
        """熵解码过程，将压缩的数据转换为DCT系数"""
        if self.switch < 2:
            return
        
        # 初始化DCT系数存储
        self.dct_coefficients = []
        
        # 重置位读取状态
        self.pos = self.scan_data_pos
        self.bit_buffer = 0
        self.bit_pos = -1
        
        # 记录日志
        self.logger.log('32', f"图像尺寸: {self.width}x{self.height}")
        self.logger.log('32', f"采样方式: {self.sampling_mode}")
        
        # 处理每个MCU
        mcu_processed = 0
        try:
            while mcu_processed < self.mcu_count:
                mcu_data = self._decode_mcu()
                self.dct_coefficients.append(mcu_data)
                mcu_processed += 1
                
                # 每处理100个MCU记录一次日志
                if mcu_processed % 100 == 0:
                    self.logger.log('32', f"已处理MCU数量: {mcu_processed}")
        except Exception as e:
            self.logger.log('32', f"解码中断于MCU {mcu_processed}: {str(e)}")
            raise
        
        self.logger.log('32', f"完成解码，总MCU数: {mcu_processed}")

    def _decode_mcu(self):
        """解码单个MCU的数据"""
        mcu_data = {}
        
        # 根据采样方式决定需要解码的块数
        if self.sampling_mode == '4:4:4':
            blocks_to_decode = {
                1: [(1, 'Y')],           # 1个Y
                2: [(1, 'Cb')],          # 1个Cb
                3: [(1, 'Cr')]           # 1个Cr
            }
        elif self.sampling_mode == '4:2:2':
            blocks_to_decode = {
                1: [(1, 'Y1'), (2, 'Y2')],  # 2个Y
                2: [(1, 'Cb')],             # 1个Cb
                3: [(1, 'Cr')]              # 1个Cr
            }
        else:  # 4:2:0
            blocks_to_decode = {
                1: [(1, 'Y1'), (2, 'Y2'), (3, 'Y3'), (4, 'Y4')],  # 4个Y
                2: [(1, 'Cb')],                                    # 1个Cb
                3: [(1, 'Cr')]                                     # 1个Cr
            }
        
        # 解码每个量的块
        for component_id, blocks in blocks_to_decode.items():
            component_info = self.scan_components[component_id]
            dc_table = self.huffman_tables['dc'][component_info['dc_table_id']]
            ac_table = self.huffman_tables['ac'][component_info['ac_table_id']]
            
            for block_num, block_name in blocks:
                # 码DC系数
                dc_value = self._decode_dc_coefficient(dc_table, component_info)
                
                # 解码AC系数
                ac_values = self._decode_ac_coefficients(ac_table)
                
                # 组合DC和AC系数
                block = self._combine_coefficients(dc_value, ac_values)
                
                # 存储解码后块
                mcu_data[block_name] = block
        
        return mcu_data

    def _decode_dc_coefficient(self, dc_table, component_info):
        """解码DC系数"""
        # 读取DC系数的大小类别
        dc_size = self._decode_huffman_value(dc_table)
        
        # 读取DC系数的实际值
        if dc_size == 0:
            dc_diff = 0
        else:
            dc_bits = self._read_bits(dc_size)
            dc_diff = self._extend_value(dc_bits, dc_size)
        
        # 计算实际的DC值（差分编码）
        dc_value = component_info['prev_dc'] + dc_diff
        component_info['prev_dc'] = dc_value
        
        return dc_value

    def _decode_ac_coefficients(self, ac_table):
        """解码AC系数"""
        ac_values = [0] * 63  # 63个AC系数
        current_pos = 0
        
        while current_pos < 63:
            # 读取AC系数的游程/大小对
            rs = self._decode_huffman_value(ac_table)
            
            # 解析游程和大小
            run_length = (rs >> 4) & 0x0F  # 高4位是游程
            ac_size = rs & 0x0F            # 低4位是大小
            
            if rs == 0x00:  # EOB标记
                break
            elif rs == 0xF0:  # ZRL标记（16个零）
                current_pos += 16
                continue
            
            # 跳过游程中的零
            current_pos += run_length
            
            # 读取AC系数的实际值
            if ac_size > 0:
                ac_bits = self._read_bits(ac_size)
                ac_value = self._extend_value(ac_bits, ac_size)
                if current_pos < 63:
                    ac_values[current_pos] = ac_value
                current_pos += 1
        
        return ac_values

    def _decode_huffman_value(self, huffman_table):
        """使用哈夫曼表解码一个值"""
        code = ""
        while True:
            bit = self._read_bits(1)
            code += str(bit)
            
            if code in huffman_table:
                return huffman_table[code]

    def _combine_coefficients(self, dc_value, ac_values):
        """将DC和AC系数组合成8x8块"""
        # 创建8x8的系数块
        block = [[0] * 8 for _ in range(8)]
        
        # 标准JPEG zigzag顺序
        zigzag_order = [
            0,  1,  5,  6, 14, 15, 27, 28,
            2,  4,  7, 13, 16, 26, 29, 42,
            3,  8, 12, 17, 25, 30, 41, 43,
            9, 11, 18, 24, 31, 40, 44, 53,
            10, 19, 23, 32, 39, 45, 52, 54,
            20, 22, 33, 38, 46, 51, 55, 60,
            21, 34, 37, 47, 50, 56, 59, 61,
            35, 36, 48, 49, 57, 58, 62, 63
        ]
        
        # 放置DC系数
        block[0][0] = dc_value
        
        # 按zigzag顺序放置AC系数
        for i, ac_value in enumerate(ac_values):
            pos = zigzag_order[i + 1]  # +1是因为跳过DC位置
            row = pos // 8
            col = pos % 8
            block[row][col] = ac_value
        
        return block

    def dequantize(self):
        """反量化过程，将量化后的DCT系数还原"""
        if self.switch < 3:
            return
        
        # 记录日志
        self.logger.log('33', f"图像尺寸: {self.width}x{self.height}")
        self.logger.log('33', f"采样方式: {self.sampling_mode}")
        
        # 初始化反量化后的DCT系数存储
        self.dequantized_coefficients = []
        
        # 处理每个MCU
        mcu_processed = 0
        try:
            for mcu_data in self.dct_coefficients:
                dequantized_mcu = {}
                
                # 根据采样方式处理不同的块
                if self.sampling_mode == '4:4:4':
                    # Y分量使用亮度量化表
                    dequantized_mcu['Y'] = self._dequantize_block(
                        mcu_data['Y'], 
                        self.quant_tables[self.sampling_factors[1]['quant_table_id']]
                    )
                    
                    # Cb和Cr分量使用色度量化表
                    dequantized_mcu['Cb'] = self._dequantize_block(
                        mcu_data['Cb'],
                        self.quant_tables[self.sampling_factors[2]['quant_table_id']]
                    )
                    dequantized_mcu['Cr'] = self._dequantize_block(
                        mcu_data['Cr'],
                        self.quant_tables[self.sampling_factors[3]['quant_table_id']]
                    )
                    
                elif self.sampling_mode == '4:2:2':
                    # 处理两个Y块
                    y_quant_table = self.quant_tables[self.sampling_factors[1]['quant_table_id']]
                    dequantized_mcu['Y1'] = self._dequantize_block(mcu_data['Y1'], y_quant_table)
                    dequantized_mcu['Y2'] = self._dequantize_block(mcu_data['Y2'], y_quant_table)
                    
                    # 处理Cb和Cr块
                    cb_quant_table = self.quant_tables[self.sampling_factors[2]['quant_table_id']]
                    cr_quant_table = self.quant_tables[self.sampling_factors[3]['quant_table_id']]
                    dequantized_mcu['Cb'] = self._dequantize_block(mcu_data['Cb'], cb_quant_table)
                    dequantized_mcu['Cr'] = self._dequantize_block(mcu_data['Cr'], cr_quant_table)
                    
                else:  # 4:2:0
                    # 处理四个Y块
                    y_quant_table = self.quant_tables[self.sampling_factors[1]['quant_table_id']]
                    dequantized_mcu['Y1'] = self._dequantize_block(mcu_data['Y1'], y_quant_table)
                    dequantized_mcu['Y2'] = self._dequantize_block(mcu_data['Y2'], y_quant_table)
                    dequantized_mcu['Y3'] = self._dequantize_block(mcu_data['Y3'], y_quant_table)
                    dequantized_mcu['Y4'] = self._dequantize_block(mcu_data['Y4'], y_quant_table)
                    
                    # 处理Cb和Cr块
                    cb_quant_table = self.quant_tables[self.sampling_factors[2]['quant_table_id']]
                    cr_quant_table = self.quant_tables[self.sampling_factors[3]['quant_table_id']]
                    dequantized_mcu['Cb'] = self._dequantize_block(mcu_data['Cb'], cb_quant_table)
                    dequantized_mcu['Cr'] = self._dequantize_block(mcu_data['Cr'], cr_quant_table)
                
                self.dequantized_coefficients.append(dequantized_mcu)
                mcu_processed += 1
                
                # 每处理100个MCU记录一次日志
                if mcu_processed % 100 == 0:
                    self.logger.log('33', f"已处理MCU数量: {mcu_processed}")
                
        except Exception as e:
            self.logger.log('33', f"反量化中断于MCU {mcu_processed}: {str(e)}")
            raise
        
        # 记录完成日志
        self.logger.log('33', f"完成反量化，总MCU数: {mcu_processed}")
        
        # 记录一些统计信息
        if mcu_processed > 0:
            # 取第一个MCU的Y分量作为示例
            sample_block = None
            if self.sampling_mode == '4:4:4':
                sample_block = self.dequantized_coefficients[0]['Y']
            else:
                sample_block = self.dequantized_coefficients[0]['Y1']
                
            if sample_block:
                # 计算示例块的统计信息
                flat_values = [val for row in sample_block for val in row]
                min_val = min(flat_values)
                max_val = max(flat_values)
                avg_val = sum(flat_values) / len(flat_values)
                self.logger.log('33', f"示例块统计: 最小值={min_val}, 最大值={max_val}, 平均值={avg_val:.2f}")

    def inverse_dct(self):
        """IDCT变换，将频率域的系数转换回空间域"""
        if self.switch < 4:
            return
        
        # 记录日志
        self.logger.log('34', f"图像尺寸: {self.width}x{self.height}")
        self.logger.log('34', f"采样方式: {self.sampling_mode}")
        
        # 初始化IDCT结果存储
        self.idct_blocks = []
        
        # 处理每个MCU
        mcu_processed = 0
        try:
            for dct_mcu in self.dequantized_coefficients:
                idct_mcu = {}
                
                # 根据采样方式处理不同的块
                if self.sampling_mode == '4:4:4':
                    # 处理Y, Cb, Cr块
                    idct_mcu['Y'] = self._apply_idct(dct_mcu['Y'])
                    idct_mcu['Cb'] = self._apply_idct(dct_mcu['Cb'])
                    idct_mcu['Cr'] = self._apply_idct(dct_mcu['Cr'])
                    
                elif self.sampling_mode == '4:2:2':
                    # 处理两个Y块
                    idct_mcu['Y1'] = self._apply_idct(dct_mcu['Y1'])
                    idct_mcu['Y2'] = self._apply_idct(dct_mcu['Y2'])
                    # 处理Cb和Cr块
                    idct_mcu['Cb'] = self._apply_idct(dct_mcu['Cb'])
                    idct_mcu['Cr'] = self._apply_idct(dct_mcu['Cr'])
                    
                else:  # 4:2:0
                    # 处理四个Y块
                    idct_mcu['Y1'] = self._apply_idct(dct_mcu['Y1'])
                    idct_mcu['Y2'] = self._apply_idct(dct_mcu['Y2'])
                    idct_mcu['Y3'] = self._apply_idct(dct_mcu['Y3'])
                    idct_mcu['Y4'] = self._apply_idct(dct_mcu['Y4'])
                    # 处理Cb和Cr块
                    idct_mcu['Cb'] = self._apply_idct(dct_mcu['Cb'])
                    idct_mcu['Cr'] = self._apply_idct(dct_mcu['Cr'])
                
                self.idct_blocks.append(idct_mcu)
                mcu_processed += 1
                
                # 每处理100个MCU记录一次日志
                if mcu_processed % 100 == 0:
                    self.logger.log('34', f"已处理MCU数量: {mcu_processed}")
                
        except Exception as e:
            self.logger.log('34', f"IDCT变换中断于MCU {mcu_processed}: {str(e)}")
            raise
        
        self.logger.log('34', f"完成IDCT变换，总MCU数: {mcu_processed}")
        
        # 记录一些统计信息
        if mcu_processed > 0:
            # 取第一个MCU的Y分量作为示例
            sample_block = None
            if self.sampling_mode == '4:4:4':
                sample_block = self.idct_blocks[0]['Y']
            else:
                sample_block = self.idct_blocks[0]['Y1']
                
            if sample_block:
                # 计算示例块的统计信息
                flat_values = [val for row in sample_block for val in row]
                min_val = min(flat_values)
                max_val = max(flat_values)
                avg_val = sum(flat_values) / len(flat_values)
                self.logger.log('34', f"示例块统计: 最小值={min_val:.2f}, 最大值={max_val:.2f}, 平均值={avg_val:.2f}")

    def _apply_idct(self, block):
        """对单个8x8块应用IDCT变换"""
        try:
            # 对行应用IDCT
            temp_block = [[0.0] * 8 for _ in range(8)]
            for i in range(8):
                temp_block[i] = idct(block[i], norm='ortho')
            
            # 对列应用IDCT
            result_block = [[0.0] * 8 for _ in range(8)]
            for j in range(8):
                column = [temp_block[i][j] for i in range(8)]
                idct_column = idct(column, norm='ortho')
                for i in range(8):
                    result_block[i][j] = idct_column[i]
            
            # 将结果限制在[0, 255]范围内
            for i in range(8):
                for j in range(8):
                    result_block[i][j] = max(0, min(255, round(result_block[i][j] + 128)))
            
            return result_block
            
        except Exception as e:
            self.logger.log('34', f"IDCT变换失败: {str(e)}")
            raise

    def color_convert(self):
        """YCbCr到RGB的颜色空间转换"""
        if self.switch < 5:
            return
        
        # 记录日志
        self.logger.log('35', f"图像尺寸: {self.width}x{self.height}")
        self.logger.log('35', f"采样方式: {self.sampling_mode}")
        
        # 初始化RGB图像数据
        self.rgb_data = []
        pixels_processed = 0
        
        try:
            # 处理每个MCU
            for mcu_index, mcu in enumerate(self.idct_blocks):
                # 计算MCU在图像中的位置
                mcu_row = (mcu_index * self.mcu_width) // self.width
                mcu_col = (mcu_index * self.mcu_width) % self.width
                
                # 根据采样方式进行转换
                if self.sampling_mode == '4:4:4':
                    # 4:4:4采样：直接一对一转换
                    rgb_mcu = self._convert_444_mcu(mcu)
                    pixels_processed += 64  # 8x8像素
                    
                elif self.sampling_mode == '4:2:2':
                    # 4:2:2采样：水平方向需要插值
                    rgb_mcu = self._convert_422_mcu(mcu)
                    pixels_processed += 128  # 16x8像素
                    
                else:  # 4:2:0
                    # 4:2:0采样：水平和垂直方向都需要插值
                    rgb_mcu = self._convert_420_mcu(mcu)
                    pixels_processed += 256  # 16x16像素
                
                self.rgb_data.append(rgb_mcu)
                
                # 每处理1000个像素记录一次日志
                if pixels_processed // 1000 > (pixels_processed - len(rgb_mcu)) // 1000:
                    self.logger.log('35', f"已处理像素数: {pixels_processed}")
                    
        except Exception as e:
            self.logger.log('35', f"颜色转换中断于MCU {mcu_index}: {str(e)}")
            raise
        
        # 记录完成日志
        total_pixels = self.width * self.height
        self.logger.log('35', f"完成颜色转换，总像素数: {total_pixels}")

    def _convert_444_mcu(self, mcu):
        """转换4:4:4采样的MCU"""
        rgb_mcu = []
        for i in range(8):
            for j in range(8):
                y = mcu['Y'][i][j]
                cb = mcu['Cb'][i][j]
                cr = mcu['Cr'][i][j]
                
                # YCbCr转RGB
                r = y + 1.402 * (cr - 128)
                g = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
                b = y + 1.772 * (cb - 128)
                
                # 限制在[0, 255]范围内
                r = max(0, min(255, round(r)))
                g = max(0, min(255, round(g)))
                b = max(0, min(255, round(b)))
                
                rgb_mcu.append((r, g, b))
        return rgb_mcu

    def _convert_422_mcu(self, mcu):
        """转换4:2:2采样的MCU"""
        rgb_mcu = []
        for i in range(8):
            for j in range(16):
                # 确定使用哪个Y块
                y_block = 'Y1' if j < 8 else 'Y2'
                y = mcu[y_block][i][j % 8]
                
                # 对Cb和Cr进行水平插值
                cb_index = j // 2
                cr_index = j // 2
                cb = mcu['Cb'][i][cb_index]
                cr = mcu['Cr'][i][cr_index]
                
                # YCbCr转RGB
                r = y + 1.402 * (cr - 128)
                g = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
                b = y + 1.772 * (cb - 128)
                
                # 限制在[0, 255]范围内
                r = max(0, min(255, round(r)))
                g = max(0, min(255, round(g)))
                b = max(0, min(255, round(b)))
                
                rgb_mcu.append((r, g, b))
        return rgb_mcu

    def _convert_420_mcu(self, mcu):
        """转换4:2:0采样的MCU"""
        rgb_mcu = []
        for i in range(16):
            for j in range(16):
                # 确定使用哪个Y块
                if i < 8:
                    y_block = 'Y1' if j < 8 else 'Y2'
                else:
                    y_block = 'Y3' if j < 8 else 'Y4'
                y = mcu[y_block][i % 8][j % 8]
                
                # 对Cb和Cr进行双线性插值
                cb_i = i // 2
                cb_j = j // 2
                cr_i = i // 2
                cr_j = j // 2
                
                cb = mcu['Cb'][cb_i][cb_j]
                cr = mcu['Cr'][cr_i][cr_j]
                
                # YCbCr转RGB
                r = y + 1.402 * (cr - 128)
                g = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
                b = y + 1.772 * (cb - 128)
                
                # 限制在[0, 255]范围内
                r = max(0, min(255, round(r)))
                g = max(0, min(255, round(g)))
                b = max(0, min(255, round(b)))
                
                rgb_mcu.append((r, g, b))
        return rgb_mcu

    def write_bmp(self):
        """将RGB数据写入BMP文件"""
        if self.switch < 6:
            return
        
        # 检查必要的属性是否已正确设置
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"无效的图像尺寸: {self.width}x{self.height}")
        
        if self.mcu_width <= 0 or self.mcu_height <= 0:
            self.logger.log('35', "MCU尺寸未正确设置，使用默认值")
            # 根据采样模式设置默认的MCU尺寸
            if self.sampling_mode == '4:4:4':
                self.mcu_width = 8
                self.mcu_height = 8
            elif self.sampling_mode == '4:2:2':
                self.mcu_width = 16
                self.mcu_height = 8
            elif self.sampling_mode == '4:2:0':
                self.mcu_width = 16
                self.mcu_height = 16
            else:
                raise ValueError(f"无效的采样模式: {self.sampling_mode}")
        
        # 计算行填充字节数（BMP要求每行字节数必须是4的倍数）
        padding = (4 - (self.width * 3) % 4) % 4
        row_size = self.width * 3 + padding
        
        # 计算文件大小
        image_size = self.height * row_size
        file_size = 54 + image_size  # 54 = 文件头(14) + 信息头(40)
        
        try:
            # 创建输出文件名
            output_filename = self.filename.rsplit('.', 1)[0] + '.bmp'
            
            with open(output_filename, 'wb') as f:
                # 1. 写入BMP文件头 (14字节)
                # 文件类型标识 'BM'
                f.write(b'BM')
                # 文件大小
                f.write(file_size.to_bytes(4, 'little'))
                # 保留字段
                f.write((0).to_bytes(4, 'little'))
                # 数据偏移量
                f.write((54).to_bytes(4, 'little'))
                
                # 2. 写入BMP信息头 (40字节)
                # 信息头大小
                f.write((40).to_bytes(4, 'little'))
                # 图像宽度
                f.write(self.width.to_bytes(4, 'little'))
                # 图像高度（负值表示从上到下存储）
                f.write((-self.height).to_bytes(4, 'little', signed=True))
                # 颜色平面数
                f.write((1).to_bytes(2, 'little'))
                # 每像素位数
                f.write((24).to_bytes(2, 'little'))
                # 压缩方式（0=不压缩）
                f.write((0).to_bytes(4, 'little'))
                # 图像数据大小
                f.write(image_size.to_bytes(4, 'little'))
                # 水平分辨率（像素/米）
                f.write((0).to_bytes(4, 'little'))
                # 垂直分辨率（像素/米）
                f.write((0).to_bytes(4, 'little'))
                # 调色板颜色数
                f.write((0).to_bytes(4, 'little'))
                # 重要颜色数
                f.write((0).to_bytes(4, 'little'))
                
                # 3. 写入图像数据
                # 将MCU数据重组为完整的图像数据
                pixels_written = 0
                mcu_cols = (self.width + self.mcu_width - 1) // self.mcu_width
                
                for y in range(self.height):
                    row_data = []
                    # 计算当前行对应的MCU行
                    mcu_row = y // self.mcu_height
                    y_in_mcu = y % self.mcu_height
                    
                    for x in range(self.width):
                        # 计算当前像素对应的MCU列和MCU内的位置
                        mcu_col = x // self.mcu_width
                        x_in_mcu = x % self.mcu_width
                        
                        # 获取对应的MCU
                        mcu_index = mcu_row * mcu_cols + mcu_col
                        if mcu_index < len(self.rgb_data):
                            mcu = self.rgb_data[mcu_index]
                            
                            # 计算像素在MCU内的索引
                            pixel_index = y_in_mcu * self.mcu_width + x_in_mcu
                            if pixel_index < len(mcu):
                                r, g, b = mcu[pixel_index]
                                # BMP格式要求BGR顺序
                                row_data.extend([b, g, r])
                                pixels_written += 1
                            else:
                                # 超出MCU范围，填充黑色
                                row_data.extend([0, 0, 0])
                        else:
                            # 超出图像范围，填充黑色
                            row_data.extend([0, 0, 0])
                    
                    # 写入行数据
                    f.write(bytes(row_data))
                    # 写入填充字节
                    f.write(bytes([0] * padding))
                    
                    # 每写入1000个像素记录一次日志
                    if pixels_written // 1000 > (pixels_written - self.width) // 1000:
                        self.logger.log('35', f"已写入像素数: {pixels_written}")
                
                # 记录完成日志
                print(f"Decoder completed: {output_filename}")
                self.logger.log('35', f"完成BMP文件写入: {output_filename}")
                self.logger.log('35', f"总像素数: {self.width * self.height}")
                self.logger.log('35', f"实际写入像素数: {pixels_written}")
                
        except Exception as e:
            self.logger.log('35', f"BMP文件写入失败: {str(e)}")
            raise

def main():
    """主函数，控制JPEG解码流程"""
    # 创建logger实例
    logger = Logger()
    
    try:
        # 要处理的文件列表
        jpeg_files = [
            "gig-sn01.jpg",
            "gig-sn08.jpg",
            "monalisa.jpg",
            "teatime.jpg"
        ]
        
        # 单文件模式
        print("\n=== 单文件模式 ===")
        # 创建解码器实例并处理第一个文件
        decoder = JPEGDecoder(jpeg_files[0])
        decoder.logger = logger
        print(f"\n开始处理: {jpeg_files[0]}")
        decoder.decode()
        print(f"完成处理: {jpeg_files[0]}")
        
        # 多文件模式
        print("\n=== 多文件模式 ===")
        # 依次处理所有文件
        for jpeg_file in jpeg_files:
            decoder = JPEGDecoder(jpeg_file)
            decoder.logger = logger
            print(f"\n开始处理: {jpeg_file}")
            decoder.decode()
            print(f"完成处理: {jpeg_file}")
        
        print("\n所有文件处理完成!")
        
    except Exception as e:
        print(f"\n程序执行出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()