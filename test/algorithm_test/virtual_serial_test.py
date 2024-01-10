import serial
import threading
import queue

def read_from_port(ser, data_queue, running):
    buffer = ''
    while running.is_set():
        data = ser.read(ser.inWaiting() or 1)
        if data:
            buffer += data.decode('utf-8')
            lines = buffer.split('\n')
            buffer = lines[-1]
            for line in lines[:-1]:
                data_queue.put(line)

def process_data(line):
    parts = line.split(',')
    processed_parts = [part.replace(':', '=') for part in parts[:-1]]
    return ','.join(processed_parts) + ','

def write_to_port(ser, data_queue, running):
    while running.is_set():
        if not data_queue.empty():
            line = data_queue.get()
            processed_line = process_data(line)
            ser.write(processed_line.encode('utf-8'))

# 配置串口
ser_real = serial.Serial('COM6', 115200)
ser_virtual = serial.Serial('COM8', 115200)

# 创建数据缓冲区
data_queue = queue.Queue()

# 创建一个事件标志来控制线程的运行
running = threading.Event()
running.set()

# 创建并启动读取和写入线程
read_thread = threading.Thread(target=read_from_port, args=(ser_real, data_queue, running))
write_thread = threading.Thread(target=write_to_port, args=(ser_virtual, data_queue, running))

read_thread.start()
write_thread.start()

try:
    while True:
        # 保持主线程运行，直到检测到键盘中断
        pass
except KeyboardInterrupt:
    # 当检测到键盘中断时，清除运行标志
    running.clear()

    # 等待线程完成
    read_thread.join()
    write_thread.join()

    # 关闭串口
    ser_real.close()
    ser_virtual.close()

    print("程序已经安全退出")
