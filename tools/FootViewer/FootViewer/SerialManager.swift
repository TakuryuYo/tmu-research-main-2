//
//  SerialManager.swift
//  FootViewer
//
//  Created by Tomo Kikuchi on 2025/06/10.
//

import Foundation
import IOKit
import IOKit.serial

protocol SerialManagerDelegate: AnyObject {
    func serialManager(_ manager: SerialManager, didReceiveData data: String)
    func serialManager(_ manager: SerialManager, didDisconnectWithError error: Error?)
}

class SerialManager: NSObject {
    weak var delegate: SerialManagerDelegate?
    
    private var fileDescriptor: Int32 = -1
    private var isConnected = false
    private var readSource: DispatchSourceRead?
    private let serialQueue = DispatchQueue(label: "jp.kikuchi.FootViewer.serial", qos: .userInteractive)
    
    var portPath: String = ""
    
    init(portPath: String) {
        self.portPath = portPath
        super.init()
    }
    
    func connect() -> Bool {
        guard !isConnected else { return true }
        
        fileDescriptor = open(portPath, O_RDWR | O_NOCTTY | O_NONBLOCK)
        
        guard fileDescriptor != -1 else {
            print("Failed to open serial port: \(portPath)")
            return false
        }
        
        // シリアルポートの設定
        var options = termios()
        tcgetattr(fileDescriptor, &options)
        
        // ボーレート設定 (115200)
        cfsetispeed(&options, speed_t(B115200))
        cfsetospeed(&options, speed_t(B115200))
        
        // 8N1設定
        options.c_cflag |= tcflag_t(CS8)
        options.c_cflag &= ~tcflag_t(PARENB)
        options.c_cflag &= ~tcflag_t(CSTOPB)
        options.c_cflag &= ~tcflag_t(CSIZE)
        options.c_cflag |= tcflag_t(CS8)
        
        // ローカルモード
        options.c_cflag |= tcflag_t(CLOCAL | CREAD)
        
        // 入力モード設定
        options.c_iflag &= ~tcflag_t(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL | IXON)
        
        // 出力モード設定
        options.c_oflag &= ~tcflag_t(OPOST)
        
        // 制御文字設定
        options.c_lflag &= ~tcflag_t(ECHO | ECHONL | ICANON | ISIG | IEXTEN)
        
        // タイムアウト設定
        withUnsafeMutablePointer(to: &options.c_cc) { cc in
            cc.pointee.16 = 0 // VMIN
            cc.pointee.17 = 1 // VTIME (0.1秒)
        }
        
        tcsetattr(fileDescriptor, TCSANOW, &options)
        
        isConnected = true
        startReading()
        
        print("Connected to serial port: \(portPath)")
        return true
    }
    
    func disconnect() {
        guard isConnected else { return }
        
        stopReading()
        
        if fileDescriptor != -1 {
            close(fileDescriptor)
            fileDescriptor = -1
        }
        
        isConnected = false
        print("Disconnected from serial port: \(portPath)")
    }
    
    private func startReading() {
        readSource = DispatchSource.makeReadSource(fileDescriptor: fileDescriptor, queue: serialQueue)
        
        readSource?.setEventHandler { [weak self] in
            self?.readData()
        }
        
        readSource?.setCancelHandler { [weak self] in
            if let fd = self?.fileDescriptor, fd != -1 {
                close(fd)
            }
        }
        
        readSource?.resume()
    }
    
    private func stopReading() {
        readSource?.cancel()
        readSource = nil
    }
    
    private func readData() {
        let bufferSize = 1024
        let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: bufferSize)
        defer { buffer.deallocate() }
        
        let bytesRead = read(fileDescriptor, buffer, bufferSize)
        
        if bytesRead > 0 {
            let data = Data(bytes: buffer, count: bytesRead)
            if let string = String(data: data, encoding: .utf8) {
                DispatchQueue.main.async {
                    self.delegate?.serialManager(self, didReceiveData: string)
                }
            }
        } else if bytesRead == -1 {
            let error = NSError(domain: NSPOSIXErrorDomain, code: Int(errno), userInfo: nil)
            DispatchQueue.main.async {
                self.delegate?.serialManager(self, didDisconnectWithError: error)
            }
        }
    }
    
    deinit {
        disconnect()
    }
}