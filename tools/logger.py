import os
import logging
import logging.handlers


class Logger:
    def __init__(self, logger_name, level, log_path, log_max_len:int=10*1024*1024, log_backup_count:int=7):
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        formatter = logging.Formatter(
            '[%(asctime)s] - [%(threadName)s %(filename)s :%(lineno)d] - %(levelname)s: %(message)s')
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(level)

        console_handle = logging.StreamHandler()
        console_handle.setFormatter(formatter)
        rotate_file_log_path = os.path.join(log_path, 'file_rotate_log')
        rotate_time_log_path = os.path.join(log_path, 'time_rotate_log')
        rotate_safe_file_log_path = os.path.join(
            log_path, 'safe_file_rotate_log')
        tar_handle = logging.handlers.RotatingFileHandler(
                        rotate_file_log_path, mode='a',
                        maxBytes=log_max_len, backupCount=log_backup_count,
                        encoding='utf-8')
        tar_handle.setFormatter(formatter)
        time_tar_handle = logging.handlers.TimedRotatingFileHandler(
            rotate_time_log_path, "D", interval=1, backupCount=log_backup_count, encoding='utf-8')
        time_tar_handle.setFormatter(formatter)
        self.logger.addHandler(console_handle)  # console
        self.logger.addHandler(tar_handle)  # 按照文件归档
        self.logger.addHandler(time_tar_handle)  # 按照时间归档
        self.logger.info("logger init")

    def get_logger(self):
        self.logger.info("get_logger")
        return self.logger
