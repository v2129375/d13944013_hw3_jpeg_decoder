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