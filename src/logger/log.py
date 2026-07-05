import sys
import threading
from datetime import datetime

LEVEL = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LEVEL2INT = {level: idx for idx, level in enumerate(LEVEL)}


class Logger(object):
    def __init__(self,
                 minimum_level: str = "DEBUG",
                 maximum_level: str = "CRITICAL",
                 log_path: str = r"./log.txt",
                 to_console: bool = True,
                 fmt: str = "[{asctime}] [{level}]: {message}",
                 terminal=sys.stdout):
        self.__check_setting(minimum_level, maximum_level, log_path)

        self.__minimum_level = LEVEL2INT[minimum_level]
        self.__maximum_level = LEVEL2INT[maximum_level]
        self.log_path = log_path
        self.to_console = to_console
        self.fmt = fmt
        self._lock = threading.Lock()
        self._initialized = False
        self.terminal = terminal
    # -------------------------------------Core Log Function-------------------------------------
    def log(self, level: str, msg: str, to_console=True, **extra):
        self.to_console = to_console
        if level not in LEVEL:
            raise ValueError(f"Unsupported log level. Supported levels: {', '.join(LEVEL)}.")
        level_index = LEVEL2INT[level]
        if not self.__minimum_level <= level_index <= self.__maximum_level:
            return None

        now = datetime.now()
        asctime = now.strftime("%Y-%m-%d %H:%M:%S")
        
        record = {"asctime": asctime, "level": level, "message": msg}
        
        with self._lock:
            if self.to_console:
                print(self.fmt.format(**record), file=self.terminal)
            try:
                with open(self.log_path, "a+") as log_file:
                    log_file.write(self.fmt.format(**record) + "\n")
            except OSError as e:
                print(f"Error writing to log file: {e}", file=self.terminal)

    # -------------------------------------Log Functions-------------------------------------
    def DEBUG_log(self, msg, to_console=True, **extra):     self.log("DEBUG", msg, to_console, **extra)
    def INFO_log(self, msg, to_console=True, **extra):      self.log("INFO", msg, to_console, **extra)
    def ERROR_log(self, msg, to_console=True, **extra):     self.log("ERROR", msg, to_console, **extra)
    def CRITICAL_log(self, msg, to_console=True, **extra):  self.log("CRITICAL", msg, to_console, **extra)
    def WARNING_log(self, msg, to_console=True, **extra):   self.log("WARNING", msg, to_console, **extra)
    
    def exception(self, msg: str = "", **extra):
        import traceback
        tb = traceback.format_exc()
        self.ERROR_log(msg or "Exception", traceback=tb, **extra)
    
    # -------------------------------------ACTIVE SETTING-------------------------------------
    def set_min_level(self, level: str): self.__minimum_level = LEVEL2INT[self.__require_level(level)]
    def set_max_level(self, level: str): self.__maximum_level = LEVEL2INT[self.__require_level(level)]
    def set_levels(self, minimum_level: str, maximum_level: str):
        self.set_min_level(minimum_level); self.set_max_level(maximum_level)
        
    def current_setting(self):
        print(f"Current Setting:\n"
              f"       Minimum Level: {LEVEL[self.__minimum_level]}\n"
              f"       Maximum Level: {LEVEL[self.__maximum_level]}\n"
              f"       Log File Path: {self.log_path}\n"
              f"       To Console: {self.to_console}\n"
              f"       Format: {self.fmt}\n"
              f"       Terminal: {self.terminal}")

    # -------------------------------------Check Setting-------------------------------------
    def __check_setting(self, minimum_level, maximum_level, log_path):
        assert minimum_level in LEVEL and maximum_level in LEVEL, \
            f"Level are not supported. Supported levels: {', '.join(LEVEL)}."
        assert type(log_path) is str, "Log path must be a string."
    def __require_level(self, level: str) -> str:
        if level not in LEVEL:
            raise ValueError(f"Unsupported log level. Supported levels: {', '.join(LEVEL)}.")
        return level