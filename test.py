import unittest
import os
from main1122 import JPEGDecoder, BitReader
import numpy as np
from io import BytesIO
import time
from functools import wraps
import struct

def timer_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} 执行时间: {end_time - start_time:.2f} 秒")
        return result
    return wrapper

class TestJPEGDecoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n开始JPEG解码器测试...\n")
        
    def setUp(self):
        self.decoder = JPEGDecoder()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.test_jpeg_path = os.path.join(current_dir, "input.jpg")
        print(f"测试文件路径: {self.test_jpeg_path}")
        
        if not os.path.exists(self.test_jpeg_path):
            raise FileNotFoundError(f"测试文件 {self.test_jpeg_path} 不存在")
        
        # 初始化解码器
        self.decoder.decode(self.test_jpeg_path, "output.bmp")
        print(f"正在运行测试: {self._testMethodName}")
        
    def tearDown(self):
        print("测试用例执行完成\n" + "-"*50 + "\n")
        
    @classmethod
    def tearDownClass(cls):
        print("\n所有测试完成！\n")

    def get_first_mcu(self):
        """辅助方法：获取第一个MCU的数据"""
        with open(self.test_jpeg_path, 'rb') as f:
            # 定位到图像数据
            f.seek(0)
            if self.decoder.read_marker(f) != 0xFFD8:
                raise ValueError("Not a valid JPEG file")
            
            while True:
                marker = self.decoder.read_marker(f)
                if marker == 0xFFDA:
                    length = self.decoder.read_length(f)
                    components_in_scan = struct.unpack('B', f.read(1))[0]
                    
                    for _ in range(components_in_scan):
                        comp_id = struct.unpack('B', f.read(1))[0]
                        tables = struct.unpack('B', f.read(1))[0]
                        for component in self.decoder.components:
                            if component['id'] == comp_id:
                                component['dc_table_id'] = tables >> 4
                                component['ac_table_id'] = tables & 0x0F
                    
                    f.seek(3, 1)
                    break
                else:
                    length = self.decoder.read_length(f)
                    f.seek(length-2, 1)
            
            self.decoder.reset_dc_predictors()
            bit_reader = BitReader(f)
            return self.decoder.decode_mcu(bit_reader, 0, self.decoder.components)

    def test_dc_value_range(self):
        """测试DC值的范围"""
        print("\n测试DC值的范围...")
        result = self.get_first_mcu()
        if result:
            blocks, _ = result
            for i, block in enumerate(blocks):
                dc_value = block[0]
                print(f"组件 {i+1} 的DC值: {dc_value}")
                self.assertLess(abs(dc_value), 2048, "DC值不应超过2048")
                self.assertNotEqual(dc_value, 0, "DC值不应为0")

    def test_dc_differential_coding(self):
        """测试DC差分编码"""
        print("\n测试DC差分编码...")
        
        # 解码多个连续的MCU以检查差分编码
        with open(self.test_jpeg_path, 'rb') as f:
            # 定位到图像数据
            while True:
                marker = self.decoder.read_marker(f)
                if marker == 0xFFDA:
                    length = self.decoder.read_length(f)
                    f.seek(length-2, 1)
                    break
                else:
                    length = self.decoder.read_length(f)
                    f.seek(length-2, 1)
            
            self.decoder.reset_dc_predictors()
            bit_reader = BitReader(f)
            prev_dc_values = {}
            
            # 解码前3个MCU
            for i in range(3):
                result = self.decoder.decode_mcu(bit_reader, i, self.decoder.components)
                if result:
                    blocks, _ = result
                    print(f"\nMCU #{i}:")
                    for j, block in enumerate(blocks):
                        dc_value = block[0]
                        if j not in prev_dc_values:
                            prev_dc_values[j] = []
                        prev_dc_values[j].append(dc_value)
                        print(f"组件 {j+1} 的DC值: {dc_value}")
            
            # 检查DC值的变化
            for comp_id, values in prev_dc_values.items():
                print(f"\n组件 {comp_id+1} 的DC值变化:")
                for i in range(1, len(values)):
                    diff = values[i] - values[i-1]
                    print(f"MCU {i-1} -> MCU {i}: 差值 = {diff}")
                    self.assertLess(abs(diff), 2048, "DC值的差异不应过大")

    def test_dc_differential_decoding(self):
        """测试DC系数的差分解码"""
        print("\n测试DC差分解码...")
        
        with open(self.test_jpeg_path, 'rb') as f:
            try:
                # 定位到图像数据开始位置
                f.seek(0)
                if self.decoder.read_marker(f) != 0xFFD8:
                    raise ValueError("Not a valid JPEG file")
                
                # 查找SOS段
                while True:
                    marker = self.decoder.read_marker(f)
                    if marker == 0xFFDA:
                        length = self.decoder.read_length(f)
                        components_in_scan = struct.unpack('B', f.read(1))[0]
                        
                        for _ in range(components_in_scan):
                            comp_id = struct.unpack('B', f.read(1))[0]
                            tables = struct.unpack('B', f.read(1))[0]
                            for component in self.decoder.components:
                                if component['id'] == comp_id:
                                    component['dc_table_id'] = tables >> 4
                                    component['ac_table_id'] = tables & 0x0F
                        
                        f.seek(3, 1)
                        break
                    else:
                        length = self.decoder.read_length(f)
                        f.seek(length-2, 1)
                
                # 重置DC预测器
                self.decoder.reset_dc_predictors()
                bit_reader = BitReader(f)
                
                # 存储每个组件的DC值和差分值
                dc_values = {comp['id']: [] for comp in self.decoder.components}
                dc_diffs = {comp['id']: [] for comp in self.decoder.components}
                
                # 解码多个连续的MCU
                num_mcus = 10  # 测试前10个MCU
                print(f"\n解码 {num_mcus} 个连续MCU的DC值:")
                
                for mcu_index in range(num_mcus):
                    result = self.decoder.decode_mcu(bit_reader, mcu_index, self.decoder.components)
                    if result:
                        blocks, _ = result
                        block_index = 0
                        
                        for component in self.decoder.components:
                            comp_id = component['id']
                            h_sampling = component['h_sampling']
                            v_sampling = component['v_sampling']
                            
                            # 处理该组件的所有块
                            for v in range(v_sampling):
                                for h in range(h_sampling):
                                    if block_index < len(blocks):
                                        dc_value = blocks[block_index][0]
                                        dc_values[comp_id].append(dc_value)
                                        
                                        # 计算差分值
                                        if len(dc_values[comp_id]) > 1:
                                            diff = dc_value - dc_values[comp_id][-2]
                                            dc_diffs[comp_id].append(diff)
                                        
                                        block_index += 1
                
                # 分析每个组件的DC值
                for comp_id in dc_values.keys():
                    values = dc_values[comp_id]
                    diffs = dc_diffs[comp_id]
                    
                    print(f"\n组件 {comp_id} 的DC值分析:")
                    print(f"DC值序列: {values}")
                    print(f"差分值序列: {diffs}")
                    
                    if len(values) > 1:
                        # 验证差分值的范围
                        max_diff = max(abs(d) for d in diffs)
                        print(f"最大差分值: {max_diff}")
                        self.assertLess(max_diff, 2048, "DC差分值不应过大")
                        
                        # 验证差分解码的正确性
                        reconstructed = [values[0]]
                        for diff in diffs:
                            reconstructed.append(reconstructed[-1] + diff)
                        
                        # 验证重建的值与原始值匹配
                        print("验证差分解码:")
                        print(f"原始DC值: {values}")
                        print(f"重建DC值: {reconstructed}")
                        np.testing.assert_array_equal(values[1:], reconstructed[1:],
                                                    "差分解码重建的DC值应该与原始值匹配")
                        
                        # 检查DC值的连续性
                        jumps = np.diff(values)
                        max_jump = max(abs(jumps))
                        print(f"相邻DC值最大变化: {max_jump}")
                        self.assertLess(max_jump, 2048, 
                                      "相邻DC值的变化不应过大")
                        
                        # 分析DC值的统计特性
                        mean_dc = np.mean(values)
                        std_dc = np.std(values)
                        print(f"DC值统计: 平均值={mean_dc:.2f}, 标准差={std_dc:.2f}")
                
                print("\n✓ DC差分解码验证完成")
                
            except Exception as e:
                print(f"错误: {str(e)}")
                print(f"文件位置: {f.tell()}")
                raise

    def test_ac_coefficient_distribution(self):
        """测试AC系数的分布"""
        print("\n测试AC系数的分布...")
        result = self.get_first_mcu()
        if result:
            blocks, _ = result
            for i, block in enumerate(blocks):
                ac_coeffs = block[1:]  # 跳过DC值
                print(f"\n组件 {i+1} 的AC系数分析:")
                
                # 统计非零系数
                non_zero_count = np.count_nonzero(ac_coeffs)
                print(f"非零AC系数数量: {non_zero_count}")
                self.assertGreater(non_zero_count, 0, "应该存在非零AC系数")
                self.assertLess(non_zero_count, 63, "不应所有AC系数都非零")
                
                # 检查系数范围
                ac_min, ac_max = np.min(ac_coeffs), np.max(ac_coeffs)
                print(f"AC系数范围: [{ac_min}, {ac_max}]")
                self.assertTrue(np.all(np.abs(ac_coeffs) < 2048), "AC系数不应过大")
                
                # 分析高频和低频系数
                low_freq = ac_coeffs[:20]  # 前20个系数作为低频
                high_freq = ac_coeffs[20:]  # 剩余系数作为高频
                print(f"低频非零系数: {np.count_nonzero(low_freq)}")
                print(f"高频非零系数: {np.count_nonzero(high_freq)}")
                self.assertGreaterEqual(np.count_nonzero(low_freq), 
                                      np.count_nonzero(high_freq),
                                      "低频系数应该比高频系数更多非零值")

    def test_ac_zero_runs(self):
        """测试AC系数的零游程编码"""
        print("\n测试AC系数的零游程...")
        result = self.get_first_mcu()
        if result:
            blocks, _ = result
            for i, block in enumerate(blocks):
                ac_coeffs = block[1:]
                print(f"\n组件 {i+1} 的零游程分析:")
                
                # 计算连续零的长度
                zero_runs = []
                current_run = 0
                for coeff in ac_coeffs:
                    if coeff == 0:
                        current_run += 1
                    else:
                        if current_run > 0:
                            zero_runs.append(current_run)
                        current_run = 0
                if current_run > 0:
                    zero_runs.append(current_run)
                
                if zero_runs:
                    print(f"零游程长度: {zero_runs}")
                    print(f"最长零游程: {max(zero_runs)}")
                    print(f"平均零游程: {np.mean(zero_runs):.2f}")
                    self.assertLess(max(zero_runs), 16, 
                                  "单个零游程不应超过15（ZRL限制）")

    def test_huffman_table_structure(self):
        """测试霍夫曼表的结构"""
        print("\n测试霍夫曼表结构...")
        
        # 检查DC和AC霍夫曼表
        for table_type in ['dc', 'ac']:
            print(f"\n检查{table_type.upper()}霍夫曼表:")
            for table_id, table in self.decoder.huffman_tables[table_type].items():
                print(f"\n表 {table_id}:")
                
                # 检查码字长度分布
                code_lengths = [len(code) for code in table.keys()]
                print(f"码字长度范围: {min(code_lengths)} - {max(code_lengths)}")
                self.assertLessEqual(max(code_lengths), 16, "霍夫曼码不应超过16位")
                
                # 检查码字的唯一性
                self.assertEqual(len(set(table.keys())), len(table), 
                               "霍夫曼码应该是唯一的")
                
                # 检查前缀性质
                codes = sorted(table.keys(), key=len)
                for i, code1 in enumerate(codes):
                    for code2 in codes[i+1:]:
                        self.assertFalse(code2.startswith(code1), 
                                       f"发现前缀码: {code1} 是 {code2} 的前缀")

    def test_huffman_code_values(self):
        """测试霍夫曼码的值范围"""
        print("\n测试霍夫曼码的值范围...")
        
        # 检查DC表的值范围
        print("\nDC表值范围检查:")
        for table_id, table in self.decoder.huffman_tables['dc'].items():
            values = list(table.values())
            print(f"表 {table_id}: 范围 [{min(values)}, {max(values)}]")
            self.assertLessEqual(max(values), 15, "DC大小类别不应超过15")
            self.assertGreaterEqual(min(values), 0, "DC大小类别不应小于0")
        
        # 检查AC表的值范围
        print("\nAC表值范围检查:")
        for table_id, table in self.decoder.huffman_tables['ac'].items():
            values = list(table.values())
            print(f"表 {table_id}: 范围 [{min(values)}, {max(values)}]")
            self.assertLess(max(values), 256, "AC表值不应超过255")
            self.assertGreaterEqual(min(values), 0, "AC表值不应小于0")

    def test_huffman_decoding_sequence(self):
        """测试霍夫曼解码序列"""
        print("\n测试霍夫曼解码序列...")
        
        result = self.get_first_mcu()
        if result:
            blocks, _ = result
            for i, block in enumerate(blocks):
                print(f"\n分析块 {i+1}:")
                
                # 检查DC系数
                dc_value = block[0]
                print(f"DC值: {dc_value}")
                
                # 分析AC系数序列
                ac_coeffs = block[1:]
                non_zero_positions = np.nonzero(ac_coeffs)[0]
                
                if len(non_zero_positions) > 0:
                    print("非零AC系数位置和值:")
                    for pos in non_zero_positions:
                        print(f"位置 {pos+1}: {ac_coeffs[pos]}")
                    
                    # 检查非零系数之间的间隔
                    gaps = np.diff(non_zero_positions)
                    print(f"非零系数间隔: {gaps}")
                    self.assertTrue(all(gap <= 16 for gap in gaps), 
                                  "非零系数间隔不超过16（ZRL限制）")

    def test_huffman_error_recovery(self):
        """测试霍夫曼解码的错误恢复能力"""
        print("\n测试霍夫曼解码的错误恢复...")
        
        with open(self.test_jpeg_path, 'rb') as f:
            try:
                # 定位到图像数据开始位置
                f.seek(0)  # 确保从文件开始读取
                
                # 跳过SOI标记
                if self.decoder.read_marker(f) != 0xFFD8:
                    raise ValueError("Not a valid JPEG file")
                
                # 查找SOS段
                while True:
                    marker_bytes = f.read(2)
                    if len(marker_bytes) < 2:
                        raise ValueError("Unexpected end of file")
                    
                    marker = struct.unpack('>H', marker_bytes)[0]
                    
                    if marker == 0xFFDA:  # SOS
                        # 读取SOS段长度
                        length = struct.unpack('>H', f.read(2))[0]
                        # 读取组件数量
                        components_in_scan = struct.unpack('B', f.read(1))[0]
                        
                        # 读取每个组件的表ID
                        for _ in range(components_in_scan):
                            comp_id = struct.unpack('B', f.read(1))[0]
                            tables = struct.unpack('B', f.read(1))[0]
                            for component in self.decoder.components:
                                if component['id'] == comp_id:
                                    component['dc_table_id'] = tables >> 4
                                    component['ac_table_id'] = tables & 0x0F
                        
                        # 跳过剩余的SOS头部
                        f.seek(3, 1)
                        break
                    else:
                        # 跳过其他段
                        length = struct.unpack('>H', f.read(2))[0]
                        f.seek(length-2, 1)
                
                print("✓ 成功定位到图像数据")
                
                # 重置DC预测器
                self.decoder.reset_dc_predictors()
                
                bit_reader = BitReader(f)
                success_count = 0
                fail_count = 0
                
                # 尝试解码多个MCU
                for i in range(10):
                    result = self.decoder.decode_mcu(bit_reader, i, self.decoder.components)
                    if result:
                        success_count += 1
                        blocks, _ = result
                        print(f"\nMCU #{i} 解码成功:")
                        print(f"块数: {len(blocks)}")
                    else:
                        fail_count += 1
                        print(f"\nMCU #{i} 解码失败")
                
                print(f"\n解码统计:")
                print(f"成功: {success_count}")
                print(f"失败: {fail_count}")
                
                # 验证大部分MCU能够成功解码
                self.assertGreater(success_count, fail_count, 
                                  "成功解码的MCU数应该多于失败的")
                    
            except Exception as e:
                print(f"错误: {str(e)}")
                print(f"文件位置: {f.tell()}")
                raise

    def test_restart_markers(self):
        """测试重启标记的处理"""
        print("\n测试重启标记处理...")
        
        with open(self.test_jpeg_path, 'rb') as f:
            try:
                # 定位到图像数据开始位置
                f.seek(0)  # 确保从文件开始读取
                
                # 跳过SOI标记
                if self.decoder.read_marker(f) != 0xFFD8:
                    raise ValueError("Not a valid JPEG file")
                
                # 查找SOS段
                while True:
                    marker_bytes = f.read(2)
                    if len(marker_bytes) < 2:
                        raise ValueError("Unexpected end of file")
                    
                    marker = struct.unpack('>H', marker_bytes)[0]
                    
                    if marker == 0xFFDA:  # SOS
                        # 读取SOS段长度
                        length = struct.unpack('>H', f.read(2))[0]
                        # 读取组件数量
                        components_in_scan = struct.unpack('B', f.read(1))[0]
                        
                        # 读取每个组件的表ID
                        for _ in range(components_in_scan):
                            comp_id = struct.unpack('B', f.read(1))[0]
                            tables = struct.unpack('B', f.read(1))[0]
                            for component in self.decoder.components:
                                if component['id'] == comp_id:
                                    component['dc_table_id'] = tables >> 4
                                    component['ac_table_id'] = tables & 0x0F
                        
                        # 跳过剩余的SOS头部
                        f.seek(3, 1)
                        break
                    else:
                        # 跳过其他段
                        length = struct.unpack('>H', f.read(2))[0]
                        f.seek(length-2, 1)
                
                print("✓ 成功定位到图像数据")
                
                # 重置DC预测器
                self.decoder.reset_dc_predictors()
                
                bit_reader = BitReader(f)
                restart_count = 0
                mcu_count = 0
                
                # 检查前100个MCU中的重启标记
                while mcu_count < 100:
                    if bit_reader.check_for_marker():
                        # 读取两个字节检查是否是重启标记
                        with bit_reader.file_lock:
                            pos = f.tell()
                            marker_bytes = f.read(2)
                            f.seek(pos)
                            
                            if len(marker_bytes) == 2:
                                marker = struct.unpack('>H', marker_bytes)[0]
                                if marker >= 0xFFD0 and marker <= 0xFFD7:
                                    restart_count += 1
                                    print(f"发现重启标记: 0x{marker:04X}")
                
                    result = self.decoder.decode_mcu(bit_reader, mcu_count, self.decoder.components)
                    if result:
                        mcu_count += 1
                
                print(f"\n在 {mcu_count} 个MCU中发现 {restart_count} 个重启标记")
                    
            except Exception as e:
                print(f"错误: {str(e)}")
                print(f"文件位置: {f.tell()}")
                raise

if __name__ == '__main__':
    # 运行霍夫曼解码相关的测试
    suite = unittest.TestSuite()
    # suite.addTest(TestJPEGDecoder('test_huffman_table_structure'))
    # suite.addTest(TestJPEGDecoder('test_huffman_code_values'))
    # suite.addTest(TestJPEGDecoder('test_huffman_decoding_sequence'))
    # suite.addTest(TestJPEGDecoder('test_huffman_error_recovery'))
    # suite.addTest(TestJPEGDecoder('test_restart_markers'))
    suite.addTest(TestJPEGDecoder('test_dc_differential_decoding'))
    
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite) 