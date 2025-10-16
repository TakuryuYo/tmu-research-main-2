//
//  DataManager.swift
//  FootViewer
//
//  Created by Tomo Kikuchi on 2025/06/10.
//

import Foundation

struct FootPressureData {
    let timestamp: String
    let rightFootData: [Double]
    let leftFootData: [Double]
    
    init(timestamp: String, rightFootData: [Double], leftFootData: [Double]) {
        self.timestamp = timestamp
        self.rightFootData = rightFootData
        self.leftFootData = leftFootData
    }
}

protocol DataManagerDelegate: AnyObject {
    func dataManager(_ manager: DataManager, didReceiveFootData data: FootPressureData)
}

class DataManager: NSObject {
    weak var delegate: DataManagerDelegate?
    
    private var csvFileURL: URL?
    private var fileHandle: FileHandle?
    private var dataBuffer: String = ""
    
    private let dateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd'T'HH-mm-ss.SSS"
        return formatter
    }()
    
    func setupCSVFile(saveDirectory: String, subjectCode: String, action: String, datasetIndex: Int) -> Bool {
        let timestamp = dateFormatter.string(from: Date())
        let filename = "\(timestamp)-\(datasetIndex.formatted(.number.precision(.fractionLength(0))))_\(subjectCode)_\(action).csv"
        
        csvFileURL = URL(fileURLWithPath: saveDirectory).appendingPathComponent(filename)
        
        guard let url = csvFileURL else { return false }
        
        // CSVヘッダーを作成
        let header = "timestamp,right_ch0,right_ch1,right_ch2,right_ch3,right_ch4,right_ch5,right_ch6,right_ch7,right_ch8,right_ch9,right_ch10,right_ch11,right_ch12,right_ch13,right_ch14,right_ch15,left_ch0,left_ch1,left_ch2,left_ch3,left_ch4,left_ch5,left_ch6,left_ch7,left_ch8,left_ch9,left_ch10,left_ch11,left_ch12,left_ch13,left_ch14,left_ch15\n"
        
        do {
            try header.write(to: url, atomically: true, encoding: .utf8)
            fileHandle = try FileHandle(forWritingTo: url)
            fileHandle?.seekToEndOfFile()
            
            print("CSV file created: \(filename)")
            return true
        } catch {
            print("Failed to create CSV file: \(error)")
            return false
        }
    }
    
    func closeCSVFile() {
        fileHandle?.closeFile()
        fileHandle = nil
        csvFileURL = nil
    }
    
    func processSerialData(_ data: String) {
        dataBuffer += data
        
        // 改行で分割してCSV行を処理
        let lines = dataBuffer.components(separatedBy: .newlines)
        
        // 最後の不完全な行を保持
        if lines.count > 1 {
            dataBuffer = lines.last ?? ""
            
            // 完全な行を処理
            for line in lines.dropLast() {
                if !line.isEmpty {
                    processCSVLine(line)
                }
            }
        }
    }
    
    private func processCSVLine(_ line: String) {
        let components = line.trimmingCharacters(in: .whitespacesAndNewlines).components(separatedBy: ",")
        
        // time, ch0, ch1, ..., ch15 (17要素) または
        // time, right_ch0-ch15, left_ch0-ch15 (33要素) の形式を想定
        
        if components.count == 17 {
            // 単一足のデータの場合
            processSingleFootData(components)
        } else if components.count == 33 {
            // 両足のデータの場合
            processBothFeetData(components)
        } else {
            print("Invalid CSV format. Expected 17 or 33 components, got \(components.count)")
            return
        }
    }
    
    private func processSingleFootData(_ components: [String]) {
        guard components.count == 17 else { return }
        
        let timestamp = components[0]
        
        // チャンネルデータを Double 配列に変換
        let channelData = components.dropFirst().compactMap { Double($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
        
        guard channelData.count == 16 else {
            print("Invalid channel data count: \(channelData.count)")
            return
        }
        
        // 正規化（0.0-1.0の範囲に変換）
        // 実際のセンサーの値の範囲に応じて調整が必要
        let normalizedData = channelData.map { max(0.0, min(1.0, $0 / 1023.0)) }
        
        // 現在の実装では右足として扱う（設定で変更可能にする予定）
        let footData = FootPressureData(
            timestamp: timestamp,
            rightFootData: normalizedData,
            leftFootData: Array(repeating: 0.0, count: 16)
        )
        
        // CSV に書き込み
        saveDataToCSV(footData)
        
        // デリゲートに通知
        delegate?.dataManager(self, didReceiveFootData: footData)
    }
    
    private func processBothFeetData(_ components: [String]) {
        guard components.count == 33 else { return }
        
        let timestamp = components[0]
        
        // 右足データ (ch0-ch15)
        let rightChannelData = Array(components[1...16]).compactMap { Double($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
        
        // 左足データ (ch16-ch31)
        let leftChannelData = Array(components[17...32]).compactMap { Double($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
        
        guard rightChannelData.count == 16 && leftChannelData.count == 16 else {
            print("Invalid channel data count: right=\(rightChannelData.count), left=\(leftChannelData.count)")
            return
        }
        
        // 正規化
        let normalizedRightData = rightChannelData.map { max(0.0, min(1.0, $0 / 1023.0)) }
        let normalizedLeftData = leftChannelData.map { max(0.0, min(1.0, $0 / 1023.0)) }
        
        let footData = FootPressureData(
            timestamp: timestamp,
            rightFootData: normalizedRightData,
            leftFootData: normalizedLeftData
        )
        
        // CSV に書き込み
        saveDataToCSV(footData)
        
        // デリゲートに通知
        delegate?.dataManager(self, didReceiveFootData: footData)
    }
    
    private func saveDataToCSV(_ data: FootPressureData) {
        guard let fileHandle = fileHandle else { return }
        
        // CSV行を作成
        var csvLine = data.timestamp
        
        // 右足データを追加
        for value in data.rightFootData {
            csvLine += ",\(value)"
        }
        
        // 左足データを追加
        for value in data.leftFootData {
            csvLine += ",\(value)"
        }
        
        csvLine += "\n"
        
        // ファイルに書き込み
        if let csvData = csvLine.data(using: .utf8) {
            fileHandle.write(csvData)
        }
    }
}