import unittest
from main import JPEGDecoder
import numpy as np

class TestJPEGDecoder(unittest.TestCase):
    def setUp(self):
        self.decoder = JPEGDecoder()

    def test_build_huffman_table(self):
        # 测试用例1：简单的霍夫曼表构建
        bits_length = [0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 1位1个码字，2位2个码字
        huffman_codes = [1, 2, 3]  # 对应的符号
        table = self.decoder._build_huffman_table(bits_length, huffman_codes)
        
        # 验证表的内容
        self.assertEqual(len(table), 3)
        self.assertEqual(table['0'], 1)  # 1位码字
        self.assertEqual(table['10'], 2)  # 2位码字
        self.assertEqual(table['11'], 3)  # 2位码字

    def test_read_entropy_coded_data(self):
        # 创建一个模拟的压缩数据
        mock_data = bytes([
            0xFF, 0x00,  # 测试填充字节
            0x12, 0x34,  # 普通数据
            0xFF, 0xD9   # EOI标记
        ])
        
        # 创建一个模拟的文件对象
        from io import BytesIO
        mock_file = BytesIO(mock_data)
        
        # 测试读取
        self.decoder._read_entropy_coded_data(mock_file)
        
        # 验证位流数据
        self.assertGreater(len(self.decoder.bits_data), 0)
        print("位流数据:", self.decoder.bits_data[:32])

    def test_decode_block_with_empty_bits(self):
        # 测试空位流情况
        self.decoder.bits_data = np.array([], dtype=np.int32)
        block = self.decoder._decode_block(0, 0, 0)
        self.assertEqual(len(block), 64)
        self.assertTrue(all(x == 0 for x in block))

    def test_decode_block_with_valid_bits(self):
        # 设置有效的位流数据
        self.decoder.bits_data = np.array([1, 0, 1, 0, 1, 1, 0, 0], dtype=np.int32)
        
        # 设置霍夫曼表
        self.decoder.huffman_tables['dc'][0] = {'101': 2}  # 示例DC表
        self.decoder.huffman_tables['ac'][0] = {'11': 0x00}  # EOB
        
        # 解码块
        block = self.decoder._decode_block(0, 0, 0)
        
        # 验证结果
        self.assertEqual(len(block), 64)
        print("解码后的块:", block[:8])  # 打印前8个值

    def test_decode_ac_coefficients(self):
        # 测试AC系数解码
        block = [0] * 64
        self.decoder.bits_data = np.array([1, 1, 0, 0], dtype=np.int32)  # EOB编码
        self.decoder.huffman_tables['ac'][0] = {'11': 0x00}  # EOB
        
        result = self.decoder._decode_ac_coefficients(0, block)
        self.assertTrue(result)
        self.assertTrue(all(x == 0 for x in block[1:]))

    def test_decode_dc_coefficient(self):
        # 测试DC系数解码
        self.decoder.bits_data = np.array([1, 0, 1, 1], dtype=np.int32)
        self.decoder.huffman_tables['dc'][0] = {'101': 1}
        
        dc_value = self.decoder._decode_dc_coefficient(0)
        print("DC解码值:", dc_value)

    def test_bits_data_management(self):
        # 测试位流数据管理
        original_bits = np.array([1, 0, 1, 0, 1, 1], dtype=np.int32)
        self.decoder.bits_data = original_bits.copy()
        
        # 测试获取位
        bits = self.decoder._get_next_bits(3)
        self.assertEqual(bits, '101')
        self.assertEqual(len(self.decoder.bits_data), 3)
        
        # 测试位流恢复
        self.decoder.bits_data = original_bits.copy()
        print("位流恢复测试:", self.decoder.bits_data)

    def test_full_decoding_process(self):
        # 测试完整的解码过程
        # 设置基本的图像参数
        self.decoder.height = 8
        self.decoder.width = 8
        
        # 设置霍夫曼表
        self.decoder.huffman_tables['dc'][0] = {'0': 0, '10': 1, '11': 2}
        self.decoder.huffman_tables['ac'][0] = {'0': 0x00, '10': 0x01, '11': 0x11}
        
        # 设置位流数据
        self.decoder.bits_data = np.array([1, 0, 1, 0, 0, 0], dtype=np.int32)
        
        # 创建扫描组件
        scan_components = [{'id': 1, 'dc_table_id': 0, 'ac_table_id': 0}]
        
        # 尝试解码
        try:
            decoded_data = self.decoder._decode_huffman_data(scan_components)
            print("解码结果:", decoded_data[0] if decoded_data else None)
        except Exception as e:
            print("解码过程出错:", str(e))

    def test_read_markers_with_compressed_data(self):
        """测试带有压缩数据的标记读取"""
        # 创建一个模拟的JPEG文件数据
        mock_data = bytes([
            0xFF, 0xD8,  # SOI
            0xFF, 0xC0,  # SOF0
            0x00, 0x11,  # 长度 17
            0x08,        # 精度 8
            0x00, 0x08,  # 高度 8
            0x00, 0x08,  # 宽度 8
            0x03,        # 组件数 3
            0x01, 0x11, 0x00,  # Y组件
            0x02, 0x11, 0x01,  # Cb组件
            0x03, 0x11, 0x01,  # Cr组件
            0xFF, 0xDA,  # SOS
            0x00, 0x0C,  # 长度 12
            0x03,        # 组件数 3
            0x01, 0x00,  # Y组件
            0x02, 0x11,  # Cb组件
            0x03, 0x11,  # Cr组件
            0x00, 0x3F, 0x00,  # Ss, Se, Ah/Al
            # 压缩数据
            0x12, 0x34, 0x56, 0x78,
            0xFF, 0xD9   # EOI
        ])
        
        # 创建模拟文件对象
        from io import BytesIO
        mock_file = BytesIO(mock_data)
        
        # 读取标记
        try:
            self.decoder.read_markers(mock_file)
            
            # 验证是否读取到压缩数据
            self.assertGreater(len(self.decoder.bits_data), 0)
            print(f"读取到的位流长度: {len(self.decoder.bits_data)}")
            print(f"位流前32位: {self.decoder.bits_data[:32]}")
            
        except Exception as e:
            self.fail(f"读取标记失败: {str(e)}")

    def test_read_scan_data(self):
        """测试扫描数据的读取"""
        # 创建模拟的扫描数据
        mock_data = bytes([
            0x00, 0x0C,  # 长度 12
            0x03,        # 组件数 3
            0x01, 0x00,  # Y组件
            0x02, 0x11,  # Cb组件
            0x03, 0x11,  # Cr组件
            0x00, 0x3F, 0x00,  # Ss, Se, Ah/Al
            # 压缩数据
            0x12, 0x34, 0x56, 0x78,
            0xFF, 0xD9   # EOI
        ])
        
        from io import BytesIO
        mock_file = BytesIO(mock_data)
        
        try:
            self.decoder._read_scan_data(mock_file)
            self.assertGreater(len(self.decoder.bits_data), 0)
            print(f"扫描数据读取后的位流长度: {len(self.decoder.bits_data)}")
        except Exception as e:
            self.fail(f"读取扫描数据失败: {str(e)}")

    def test_read_entropy_coded_data_with_markers(self):
        """测试带有标记的熵编码数据读取"""
        # 创建包含各种标记的压缩数据
        mock_data = bytes([
            0x12, 0x34,        # 普通数据
            0xFF, 0x00,        # 填充字节
            0x56, 0x78,        # 普通数据
            0xFF, 0xD0,        # 重启标记
            0x9A, 0xBC,        # 普通数据
            0xFF, 0xD9         # EOI标记
        ])
        
        from io import BytesIO
        mock_file = BytesIO(mock_data)
        
        try:
            self.decoder._read_entropy_coded_data(mock_file)
            self.assertGreater(len(self.decoder.bits_data), 0)
            print(f"熵编码数据读取后的位流长度: {len(self.decoder.bits_data)}")
            print(f"位流内容: {self.decoder.bits_data[:32]}")
        except Exception as e:
            self.fail(f"读取熵编码数据失败: {str(e)}")

    def test_read_markers_sequence(self):
        """测试完整的标记序列读取"""
        # 创建一个完整的JPEG文件结构
        mock_data = bytes([
            0xFF, 0xD8,        # SOI
            0xFF, 0xE0,        # APP0
            0x00, 0x10,        # 长度 16
            0x4A, 0x46, 0x49, 0x46, 0x00,  # JFIF\0
            0x01, 0x01,        # 版本 1.1
            0x00,              # 单位
            0x00, 0x01,        # X密度
            0x00, 0x01,        # Y密度
            0x00, 0x00,        # 缩略图
            0xFF, 0xDB,        # DQT
            0x00, 0x43,        # 长度 67
            0x00,              # 表ID
            *([0x10] * 64),    # 量化值
            0xFF, 0xC0,        # SOF0
            0x00, 0x11,        # 长度 17
            0x08,              # 精度
            0x00, 0x08,        # 高度
            0x00, 0x08,        # 宽度
            0x03,              # 组件数
            0x01, 0x11, 0x00,  # Y
            0x02, 0x11, 0x01,  # Cb
            0x03, 0x11, 0x01,  # Cr
            0xFF, 0xDA,        # SOS
            0x00, 0x0C,        # 长度
            0x03,              # 组件数
            0x01, 0x00,        # Y
            0x02, 0x11,        # Cb
            0x03, 0x11,        # Cr
            0x00, 0x3F, 0x00,  # Ss, Se, Ah/Al
            # 压缩数据
            0x12, 0x34, 0x56, 0x78,
            0xFF, 0xD9         # EOI
        ])
        
        from io import BytesIO
        mock_file = BytesIO(mock_data)
        
        try:
            self.decoder.read_markers(mock_file)
            
            # 验证各种数据是否正确读取
            self.assertGreater(len(self.decoder.bits_data), 0, "没有读取到压缩数据")
            self.assertTrue(self.decoder.components, "没有读取到组件信息")
            self.assertTrue(self.decoder.quantization_tables, "没有读取到量化表")
            
            print("\n标记读取测试结果:")
            print(f"位流长度: {len(self.decoder.bits_data)}")
            print(f"组件数量: {len(self.decoder.components)}")
            print(f"量化表数量: {len(self.decoder.quantization_tables)}")
            print(f"霍夫曼表数量: DC={len(self.decoder.huffman_tables['dc'])}, "
                  f"AC={len(self.decoder.huffman_tables['ac'])}")
            
        except Exception as e:
            self.fail(f"标记读取失败: {str(e)}")

if __name__ == '__main__':
    unittest.main(verbosity=2) 