//
//  ControlPanelView.swift
//  FootViewer
//
//  Created by Tomo Kikuchi on 2025/05/29.
//

import Foundation
import Cocoa
import IOKit
import IOKit.serial

protocol FootDataDelegate: AnyObject {
    func didReceiveFootData(rightFoot: [Double], leftFoot: [Double])
}

// MARK: - ViewModel
public class ControlPanelViewModel: ObservableObject {
    @Published public var isMeasuring: Bool = false
    @Published public var elapsedMilliseconds: Int = 0  // ミリ秒で管理
    @Published public var selectedAction: String = ""
    @Published public var datasetIndex: Int = 0
    @Published public var measurementCount: Int = 1
    @Published public var subjectCode: String = ""
    @Published public var measurementDuration: String = "10"
    @Published public var saveLocation: String = ""
    @Published public var rightFootSerialPort: String = ""
    @Published public var leftFootSerialPort: String = ""
    @Published public var availableSerialPorts: [String] = []
    @Published public var rightFootEnabled: Bool = true
    @Published public var leftFootEnabled: Bool = true
    @Published public var connectionStatus: String = "未接続"
    
    public let actionOptions = ["歩行", "立位", "座位", "その他"]
    public let subjectOptions = ["被験者A", "被験者B", "被験者C", "被験者D"]
    public let durationOptions = ["10", "60", "120", "240", "指定なし"]
    
    private var timer: Timer?
    private var rightFootSerialManager: SerialManager?
    private var leftFootSerialManager: SerialManager?
    private var dataManager: DataManager?
    
    weak var footDataDelegate: FootDataDelegate?
    
    public init() {
        selectedAction = actionOptions.first ?? ""
        subjectCode = subjectOptions.first ?? ""
        refreshSerialPorts()
        
        dataManager = DataManager()
        dataManager?.delegate = self
    }
    
    public func refreshSerialPorts() {
        availableSerialPorts = getAvailableSerialPorts()
        
        // デフォルト値の設定（利用可能なポートがある場合）
        if !availableSerialPorts.isEmpty {
            if rightFootSerialPort.isEmpty {
                rightFootSerialPort = availableSerialPorts.first ?? ""
            }
            if leftFootSerialPort.isEmpty && availableSerialPorts.count > 1 {
                leftFootSerialPort = availableSerialPorts[1]
            } else if leftFootSerialPort.isEmpty {
                leftFootSerialPort = availableSerialPorts.first ?? ""
            }
        }
    }
    
    private func getAvailableSerialPorts() -> [String] {
        var ports: [String] = []
        
        let matchingDict = IOServiceMatching(kIOSerialBSDServiceValue)
        var serialPortIterator: io_iterator_t = 0
        
        let kernResult = IOServiceGetMatchingServices(kIOMainPortDefault, matchingDict, &serialPortIterator)
        if kernResult == KERN_SUCCESS {
            var serialService: io_object_t
            repeat {
                serialService = IOIteratorNext(serialPortIterator)
                if serialService != 0 {
                    let key: CFString = "IOCalloutDevice" as CFString
                    let bsdPathAsCFString = IORegistryEntryCreateCFProperty(serialService, key, kCFAllocatorDefault, 0)
                    if let bsdPath = bsdPathAsCFString?.takeRetainedValue() as? String {
                        ports.append(bsdPath)
                    }
                    IOObjectRelease(serialService)
                }
            } while serialService != 0
        }
        IOObjectRelease(serialPortIterator)
        
        // ソートして一貫した順序にする
        return ports.sorted()
    }
    
    public func connectToSerialPorts() {
        disconnectSerialPorts()
        
        var connectionCount = 0
        
        if rightFootEnabled && !rightFootSerialPort.isEmpty {
            rightFootSerialManager = SerialManager(portPath: rightFootSerialPort)
            rightFootSerialManager?.delegate = self
            if rightFootSerialManager?.connect() == true {
                connectionCount += 1
            }
        }
        
        if leftFootEnabled && !leftFootSerialPort.isEmpty {
            leftFootSerialManager = SerialManager(portPath: leftFootSerialPort)
            leftFootSerialManager?.delegate = self
            if leftFootSerialManager?.connect() == true {
                connectionCount += 1
            }
        }
        
        if connectionCount > 0 {
            connectionStatus = "\(connectionCount)台接続済み"
        } else {
            connectionStatus = "接続失敗"
        }
    }
    
    public func disconnectSerialPorts() {
        rightFootSerialManager?.disconnect()
        leftFootSerialManager?.disconnect()
        rightFootSerialManager = nil
        leftFootSerialManager = nil
        connectionStatus = "未接続"
    }
    
    public func startMeasurement() {
        guard !saveLocation.isEmpty else {
            print("保存先が設定されていません")
            return
        }
        
        // CSV ファイルを作成
        if dataManager?.setupCSVFile(
            saveDirectory: saveLocation,
            subjectCode: subjectCode,
            action: selectedAction,
            datasetIndex: datasetIndex
        ) == true {
            isMeasuring = true
            elapsedMilliseconds = 0
            timer = Timer.scheduledTimer(withTimeInterval: 0.01, repeats: true) { _ in  // 10ms間隔
                self.elapsedMilliseconds += 10
            }
        }
    }
    
    public func stopMeasurement() {
        isMeasuring = false
        timer?.invalidate()
        timer = nil
        dataManager?.closeCSVFile()
    }
    
    public func toggleMeasurement() {
        if isMeasuring {
            stopMeasurement()
        } else {
            startMeasurement()
        }
    }
    
    // HH:mm:ss.sss形式の時間文字列を生成
    public var formattedTime: String {
        let totalMs = elapsedMilliseconds
        let hours = totalMs / (1000 * 60 * 60)
        let minutes = (totalMs % (1000 * 60 * 60)) / (1000 * 60)
        let seconds = (totalMs % (1000 * 60)) / 1000
        let milliseconds = totalMs % 1000
        
        return String(format: "%02d:%02d:%02d.%03d", hours, minutes, seconds, milliseconds)
    }
}

// MARK: - View
public final class ControlPanelView: NSView {
    public let viewModel = ControlPanelViewModel()
    
    // MARK: - UI Components
    private lazy var statusView: StatusView = StatusView()
    private lazy var timeLabel: NSTextField = generateTimeLabel()
    
    private lazy var actionLabel: NSTextField = generateLabel(text: "動作")
    private lazy var actionPopUpButton: NSPopUpButton = generateActionPopUpButton()
    
    private lazy var datasetLabel: NSTextField = generateLabel(text: "データセット")
    private lazy var datasetTextField: NSTextField = generateDatasetTextField()
    
    private lazy var measurementCountLabel: NSTextField = generateLabel(text: "測定回数")
    private lazy var measurementCountTextField: NSTextField = generateMeasurementCountTextField()
    
    private lazy var subjectLabel: NSTextField = generateLabel(text: "被験者コード")
    private lazy var subjectPopUpButton: NSPopUpButton = generateSubjectPopUpButton()
    
    private lazy var durationLabel: NSTextField = generateLabel(text: "測定秒数")
    private lazy var durationPopUpButton: NSPopUpButton = generateDurationPopUpButton()
    
    private lazy var rightFootSerialLabel: NSTextField = generateLabel(text: "ポート(右足)")
    private lazy var rightFootSerialPopUpButton: NSPopUpButton = generateRightFootSerialPopUpButton()
    private lazy var rightFootEnabledCheckBox: NSButton = generateRightFootEnabledCheckBox()
    
    private lazy var leftFootSerialLabel: NSTextField = generateLabel(text: "ポート(左足)")
    private lazy var leftFootSerialPopUpButton: NSPopUpButton = generateLeftFootSerialPopUpButton()
    private lazy var leftFootEnabledCheckBox: NSButton = generateLeftFootEnabledCheckBox()
    
    private lazy var refreshSerialButton: NSButton = generateRefreshSerialButton()
    private lazy var connectionStatusLabel: NSTextField = generateConnectionStatusLabel()
    
    private lazy var saveLocationLabel: NSTextField = generateLabel(text: "保存場所")
    private lazy var saveLocationButton: NSButton = generateSaveLocationButton()
    
    private lazy var measurementButton: NSButton = generateMeasurementButton()
    
    public override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        setup()
    }
    
    public required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    private func setup() {
        addSubviews()
        layoutStatusViews()
        layoutControlViews()
        layoutMeasurementButton()
        
        // ViewModelの監視
        setupBindings()
        
        // シリアルポートのポップアップを初期化
        updateSerialPortPopUps()
    }
    
    private func addSubviews() {
        addSubview(statusView)
        addSubview(timeLabel)
        addSubview(actionLabel)
        addSubview(actionPopUpButton)
        addSubview(datasetLabel)
        addSubview(datasetTextField)
        addSubview(measurementCountLabel)
        addSubview(measurementCountTextField)
        addSubview(subjectLabel)
        addSubview(subjectPopUpButton)
        addSubview(durationLabel)
        addSubview(durationPopUpButton)
        addSubview(rightFootSerialLabel)
        addSubview(rightFootSerialPopUpButton)
        addSubview(rightFootEnabledCheckBox)
        addSubview(leftFootSerialLabel)
        addSubview(leftFootSerialPopUpButton)
        addSubview(leftFootEnabledCheckBox)
        addSubview(refreshSerialButton)
        addSubview(connectionStatusLabel)
        addSubview(saveLocationLabel)
        addSubview(saveLocationButton)
        addSubview(measurementButton)
    }
    
    private func setupBindings() {
        // タイマー更新の監視
        Timer.scheduledTimer(withTimeInterval: 0.01, repeats: true) { _ in  // より高頻度で更新
            DispatchQueue.main.async {
                self.updateStatusDisplay()
            }
        }
    }
    
    private func updateStatusDisplay() {
        if viewModel.isMeasuring {
            statusView.text = "測定中"
            statusView.backgroundColor = NSColor.systemRed
            timeLabel.stringValue = viewModel.formattedTime
        } else {
            statusView.text = "停止"
            statusView.backgroundColor = NSColor.systemGray
            timeLabel.stringValue = "00:00:00.000"
        }
        
        // 接続状態を更新
        connectionStatusLabel.stringValue = viewModel.connectionStatus
    }
    
    private func updateSerialPortPopUps() {
        // 右足シリアルポート
        rightFootSerialPopUpButton.removeAllItems()
        if viewModel.availableSerialPorts.isEmpty {
            rightFootSerialPopUpButton.addItem(withTitle: "利用可能なポートなし")
        } else {
            rightFootSerialPopUpButton.addItems(withTitles: viewModel.availableSerialPorts)
            if !viewModel.rightFootSerialPort.isEmpty,
               let index = viewModel.availableSerialPorts.firstIndex(of: viewModel.rightFootSerialPort) {
                rightFootSerialPopUpButton.selectItem(at: index)
            }
        }
        rightFootSerialPopUpButton.isEnabled = viewModel.rightFootEnabled
        
        // 左足シリアルポート
        leftFootSerialPopUpButton.removeAllItems()
        if viewModel.availableSerialPorts.isEmpty {
            leftFootSerialPopUpButton.addItem(withTitle: "利用可能なポートなし")
        } else {
            leftFootSerialPopUpButton.addItems(withTitles: viewModel.availableSerialPorts)
            if !viewModel.leftFootSerialPort.isEmpty,
               let index = viewModel.availableSerialPorts.firstIndex(of: viewModel.leftFootSerialPort) {
                leftFootSerialPopUpButton.selectItem(at: index)
            }
        }
        leftFootSerialPopUpButton.isEnabled = viewModel.leftFootEnabled
    }
}

// MARK: - Custom Status View
private class StatusView: NSView {
    private let label = NSTextField()
    
    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        setupLabel()
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupLabel()
    }
    
    private func setupLabel() {
        label.isEditable = false
        label.isBordered = false
        label.isSelectable = false
        label.stringValue = "停止"
        label.font = NSFont.boldSystemFont(ofSize: 18)
        label.textColor = NSColor.white
        label.backgroundColor = NSColor.clear
        label.alignment = .center
        label.drawsBackground = false
        label.translatesAutoresizingMaskIntoConstraints = false
        
        addSubview(label)
        
        NSLayoutConstraint.activate([
            label.centerXAnchor.constraint(equalTo: centerXAnchor),
            label.centerYAnchor.constraint(equalTo: centerYAnchor)
        ])
        
        wantsLayer = true
        layer?.cornerRadius = 8
        layer?.backgroundColor = NSColor.systemGray.cgColor
    }
    
    var text: String {
        get { label.stringValue }
        set { label.stringValue = newValue }
    }
    
    var backgroundColor: NSColor? {
        get {
            if let cgColor = layer?.backgroundColor {
                return NSColor(cgColor: cgColor)
            }
            return nil
        }
        set { layer?.backgroundColor = newValue?.cgColor }
    }
}
    
// MARK: - UI Generation
extension ControlPanelView {
    private func generateTimeLabel() -> NSTextField {
        let label = NSTextField()
        label.isEditable = false
        label.isBordered = false
        label.isSelectable = false
        label.stringValue = "00:00:00.000"
        label.font = NSFont.monospacedDigitSystemFont(ofSize: 26, weight: .medium)
        label.textColor = NSColor.labelColor
        label.backgroundColor = NSColor.clear
        label.alignment = .center
        
        // 縦方向の中央揃えのための設定
        if let cell = label.cell as? NSTextFieldCell {
            cell.usesSingleLineMode = true
            cell.lineBreakMode = .byClipping
            cell.truncatesLastVisibleLine = true
        }
        
        label.translatesAutoresizingMaskIntoConstraints = false
        return label
    }
    
    private func generateLabel(text: String) -> NSTextField {
        let label = NSTextField()
        label.isEditable = false
        label.isBordered = false
        label.isSelectable = false
        label.stringValue = text
        label.font = NSFont.systemFont(ofSize: 13)
        label.textColor = NSColor.labelColor
        label.backgroundColor = NSColor.clear
        label.drawsBackground = false
        label.alignment = .right
        label.translatesAutoresizingMaskIntoConstraints = false
        return label
    }
    
    private func generateActionPopUpButton() -> NSPopUpButton {
        let popUp = NSPopUpButton()
        popUp.removeAllItems()
        popUp.addItems(withTitles: viewModel.actionOptions)
        popUp.target = self
        popUp.action = #selector(actionChanged(_:))
        popUp.translatesAutoresizingMaskIntoConstraints = false
        return popUp
    }
    
    private func generateDatasetTextField() -> NSTextField {
        let textField = NSTextField()
        textField.stringValue = "\(viewModel.datasetIndex)"
        textField.target = self
        textField.action = #selector(datasetIndexChanged(_:))
        textField.translatesAutoresizingMaskIntoConstraints = false
        return textField
    }
    
    private func generateMeasurementCountTextField() -> NSTextField {
        let textField = NSTextField()
        textField.stringValue = "\(viewModel.measurementCount)"
        textField.target = self
        textField.action = #selector(measurementCountChanged(_:))
        textField.translatesAutoresizingMaskIntoConstraints = false
        return textField
    }
    
    private func generateSubjectPopUpButton() -> NSPopUpButton {
        let popUp = NSPopUpButton()
        popUp.removeAllItems()
        popUp.addItems(withTitles: viewModel.subjectOptions)
        popUp.target = self
        popUp.action = #selector(subjectChanged(_:))
        popUp.translatesAutoresizingMaskIntoConstraints = false
        return popUp
    }
    
    private func generateDurationPopUpButton() -> NSPopUpButton {
        let popUp = NSPopUpButton()
        popUp.removeAllItems()
        popUp.addItems(withTitles: viewModel.durationOptions)
        popUp.target = self
        popUp.action = #selector(durationChanged(_:))
        popUp.translatesAutoresizingMaskIntoConstraints = false
        return popUp
    }
    
    private func generateRightFootSerialPopUpButton() -> NSPopUpButton {
        let popUp = NSPopUpButton()
        popUp.target = self
        popUp.action = #selector(rightFootSerialChanged(_:))
        popUp.translatesAutoresizingMaskIntoConstraints = false
        return popUp
    }
    
    private func generateLeftFootSerialPopUpButton() -> NSPopUpButton {
        let popUp = NSPopUpButton()
        popUp.target = self
        popUp.action = #selector(leftFootSerialChanged(_:))
        popUp.translatesAutoresizingMaskIntoConstraints = false
        return popUp
    }
    
    private func generateRightFootEnabledCheckBox() -> NSButton {
        let checkBox = NSButton(checkboxWithTitle: "有効", target: self, action: #selector(rightFootEnabledChanged(_:)))
        checkBox.state = viewModel.rightFootEnabled ? .on : .off
        checkBox.translatesAutoresizingMaskIntoConstraints = false
        return checkBox
    }
    
    private func generateLeftFootEnabledCheckBox() -> NSButton {
        let checkBox = NSButton(checkboxWithTitle: "有効", target: self, action: #selector(leftFootEnabledChanged(_:)))
        checkBox.state = viewModel.leftFootEnabled ? .on : .off
        checkBox.translatesAutoresizingMaskIntoConstraints = false
        return checkBox
    }
    
    private func generateRefreshSerialButton() -> NSButton {
        let button = NSButton()
        button.title = "更新"
        button.target = self
        button.action = #selector(refreshSerialPorts(_:))
        button.translatesAutoresizingMaskIntoConstraints = false
        return button
    }
    
    private func generateConnectionStatusLabel() -> NSTextField {
        let label = NSTextField()
        label.isEditable = false
        label.isBordered = false
        label.isSelectable = false
        label.stringValue = "未接続"
        label.font = NSFont.systemFont(ofSize: 12)
        label.textColor = NSColor.secondaryLabelColor
        label.backgroundColor = NSColor.clear
        label.drawsBackground = false
        label.alignment = .center
        label.translatesAutoresizingMaskIntoConstraints = false
        return label
    }
    
    private func generateSaveLocationButton() -> NSButton {
        let button = NSButton()
        button.title = "選択..."
        button.target = self
        button.action = #selector(selectSaveLocation(_:))
        button.translatesAutoresizingMaskIntoConstraints = false
        return button
    }
    
    private func generateMeasurementButton() -> NSButton {
        let button = NSButton(title: "測定開始", target: self, action: #selector(toggleMeasurement(_:)))
        button.translatesAutoresizingMaskIntoConstraints = false
        
        // ボタンのスタイル設定
        button.bezelStyle = .rounded
        button.contentTintColor = NSColor.controlAccentColor
        button.bezelColor = NSColor.controlAccentColor
        
        // ボタン内のテキストを中央揃え
        if let cell = button.cell as? NSButtonCell {
            cell.imagePosition = .noImage
            cell.alignment = .center
        }
        
        button.wantsLayer = true
        button.layer?.cornerRadius = 6
        
        return button
    }
}

// MARK: - Layout
extension ControlPanelView {
    private func layoutStatusViews() {
        statusView.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            // ステータスビュー（横幅いっぱい）
            statusView.topAnchor.constraint(equalTo: topAnchor, constant: 15),
            statusView.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 15),
            statusView.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -15),
            statusView.heightAnchor.constraint(equalToConstant: 50),
            
            // 時間表示（中央、高さ増加）
            timeLabel.topAnchor.constraint(equalTo: statusView.bottomAnchor, constant: 15),
            timeLabel.centerXAnchor.constraint(equalTo: centerXAnchor),
            timeLabel.widthAnchor.constraint(equalToConstant: 200),
            timeLabel.heightAnchor.constraint(equalToConstant: 30),
        ])
    }
    
    private func layoutControlViews() {
        let topOffset: CGFloat = 130
        let rowHeight: CGFloat = 28  // パディングを小さく
        let labelWidth: CGFloat = 80  // ラベル幅を少し拡張（シリアルポート用）
        
        NSLayoutConstraint.activate([
            // 動作の選択
            actionLabel.topAnchor.constraint(equalTo: topAnchor, constant: topOffset),
            actionLabel.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 15),
            actionLabel.widthAnchor.constraint(equalToConstant: labelWidth),
            
            actionPopUpButton.centerYAnchor.constraint(equalTo: actionLabel.centerYAnchor),
            actionPopUpButton.leadingAnchor.constraint(equalTo: actionLabel.trailingAnchor, constant: 8),
            actionPopUpButton.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -15),
            
            // データセットインデックス
            datasetLabel.topAnchor.constraint(equalTo: actionLabel.bottomAnchor, constant: rowHeight),
            datasetLabel.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 15),
            datasetLabel.widthAnchor.constraint(equalToConstant: labelWidth),
            
            datasetTextField.centerYAnchor.constraint(equalTo: datasetLabel.centerYAnchor),
            datasetTextField.leadingAnchor.constraint(equalTo: datasetLabel.trailingAnchor, constant: 8),
            datasetTextField.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -15),
            
            // 測定回数
            measurementCountLabel.topAnchor.constraint(equalTo: datasetLabel.bottomAnchor, constant: rowHeight),
            measurementCountLabel.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 15),
            measurementCountLabel.widthAnchor.constraint(equalToConstant: labelWidth),
            
            measurementCountTextField.centerYAnchor.constraint(equalTo: measurementCountLabel.centerYAnchor),
            measurementCountTextField.leadingAnchor.constraint(equalTo: measurementCountLabel.trailingAnchor, constant: 8),
            measurementCountTextField.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -15),
            
            // 被験者コード
            subjectLabel.topAnchor.constraint(equalTo: measurementCountLabel.bottomAnchor, constant: rowHeight),
            subjectLabel.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 15),
            subjectLabel.widthAnchor.constraint(equalToConstant: labelWidth),
            
            subjectPopUpButton.centerYAnchor.constraint(equalTo: subjectLabel.centerYAnchor),
            subjectPopUpButton.leadingAnchor.constraint(equalTo: subjectLabel.trailingAnchor, constant: 8),
            subjectPopUpButton.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -15),
            
            // 測定秒数
            durationLabel.topAnchor.constraint(equalTo: subjectLabel.bottomAnchor, constant: rowHeight),
            durationLabel.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 15),
            durationLabel.widthAnchor.constraint(equalToConstant: labelWidth),
            
            durationPopUpButton.centerYAnchor.constraint(equalTo: durationLabel.centerYAnchor),
            durationPopUpButton.leadingAnchor.constraint(equalTo: durationLabel.trailingAnchor, constant: 8),
            durationPopUpButton.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -15),
            
            // シリアルポート(右足)
            rightFootSerialLabel.topAnchor.constraint(equalTo: durationLabel.bottomAnchor, constant: rowHeight),
            rightFootSerialLabel.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 15),
            rightFootSerialLabel.widthAnchor.constraint(equalToConstant: labelWidth),
            
            rightFootEnabledCheckBox.centerYAnchor.constraint(equalTo: rightFootSerialLabel.centerYAnchor),
            rightFootEnabledCheckBox.leadingAnchor.constraint(equalTo: rightFootSerialLabel.trailingAnchor, constant: 8),
            rightFootEnabledCheckBox.widthAnchor.constraint(equalToConstant: 60),
                       
            rightFootSerialPopUpButton.centerYAnchor.constraint(equalTo: rightFootSerialLabel.centerYAnchor),
            rightFootSerialPopUpButton.leadingAnchor.constraint(equalTo: rightFootEnabledCheckBox.trailingAnchor, constant: 0),
            rightFootSerialPopUpButton.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -15),
            
            // シリアルポート(左足)
            leftFootSerialLabel.topAnchor.constraint(equalTo: rightFootSerialLabel.bottomAnchor, constant: rowHeight),
            leftFootSerialLabel.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 15),
            leftFootSerialLabel.widthAnchor.constraint(equalToConstant: labelWidth),
            
            leftFootEnabledCheckBox.centerYAnchor.constraint(equalTo: leftFootSerialLabel.centerYAnchor),
            leftFootEnabledCheckBox.leadingAnchor.constraint(equalTo: leftFootSerialLabel.trailingAnchor, constant: 8),
            leftFootEnabledCheckBox.widthAnchor.constraint(equalToConstant: 60),
            
            leftFootSerialPopUpButton.centerYAnchor.constraint(equalTo: leftFootSerialLabel.centerYAnchor),
            leftFootSerialPopUpButton.leadingAnchor.constraint(equalTo: leftFootEnabledCheckBox.trailingAnchor, constant: 0),
            leftFootSerialPopUpButton.widthAnchor.constraint(equalToConstant: 180),
            
            // シリアルポート更新ボタン
            refreshSerialButton.centerYAnchor.constraint(equalTo: leftFootSerialLabel.centerYAnchor),
            refreshSerialButton.leadingAnchor.constraint(equalTo: leftFootSerialPopUpButton.trailingAnchor, constant: 8),
            refreshSerialButton.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -15),
            refreshSerialButton.widthAnchor.constraint(equalToConstant: 60),
            
            // 接続状態表示
            connectionStatusLabel.topAnchor.constraint(equalTo: leftFootSerialLabel.bottomAnchor, constant: 8),
            connectionStatusLabel.centerXAnchor.constraint(equalTo: centerXAnchor),
            connectionStatusLabel.heightAnchor.constraint(equalToConstant: 20),
            
            // 保存場所
            saveLocationLabel.topAnchor.constraint(equalTo: connectionStatusLabel.bottomAnchor, constant: rowHeight),
            saveLocationLabel.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 15),
            saveLocationLabel.widthAnchor.constraint(equalToConstant: labelWidth),
            
            saveLocationButton.centerYAnchor.constraint(equalTo: saveLocationLabel.centerYAnchor),
            saveLocationButton.leadingAnchor.constraint(equalTo: saveLocationLabel.trailingAnchor, constant: 8),
            saveLocationButton.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -15),
        ])
    }
    
    private func layoutMeasurementButton() {
        NSLayoutConstraint.activate([
            measurementButton.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -15),
            measurementButton.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 15),
            measurementButton.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -15),
            measurementButton.topAnchor.constraint(equalTo: saveLocationButton.bottomAnchor, constant: 120)
        ])
    }
}

// MARK: - Actions
extension ControlPanelView {
    @objc private func actionChanged(_ sender: NSPopUpButton) {
        if let selectedItem = sender.selectedItem {
            viewModel.selectedAction = selectedItem.title
        }
    }
    
    @objc private func datasetIndexChanged(_ sender: NSTextField) {
        viewModel.datasetIndex = sender.integerValue
    }
    
    @objc private func measurementCountChanged(_ sender: NSTextField) {
        viewModel.measurementCount = sender.integerValue
    }
    
    @objc private func subjectChanged(_ sender: NSPopUpButton) {
        if let selectedItem = sender.selectedItem {
            viewModel.subjectCode = selectedItem.title
        }
    }
    
    @objc private func durationChanged(_ sender: NSPopUpButton) {
        if let selectedItem = sender.selectedItem {
            viewModel.measurementDuration = selectedItem.title
        }
    }
    
    @objc private func rightFootSerialChanged(_ sender: NSPopUpButton) {
        if let selectedItem = sender.selectedItem,
           viewModel.availableSerialPorts.contains(selectedItem.title) {
            viewModel.rightFootSerialPort = selectedItem.title
            // ポート変更時に自動接続
            if viewModel.rightFootEnabled {
                viewModel.connectToSerialPorts()
            }
        }
    }
    
    @objc private func leftFootSerialChanged(_ sender: NSPopUpButton) {
        if let selectedItem = sender.selectedItem,
           viewModel.availableSerialPorts.contains(selectedItem.title) {
            viewModel.leftFootSerialPort = selectedItem.title
            // ポート変更時に自動接続
            if viewModel.leftFootEnabled {
                viewModel.connectToSerialPorts()
            }
        }
    }
    
    @objc private func rightFootEnabledChanged(_ sender: NSButton) {
        viewModel.rightFootEnabled = sender.state == .on
        rightFootSerialPopUpButton.isEnabled = viewModel.rightFootEnabled
        // 有効/無効変更時に接続状態を更新
        viewModel.connectToSerialPorts()
    }
    
    @objc private func leftFootEnabledChanged(_ sender: NSButton) {
        viewModel.leftFootEnabled = sender.state == .on
        leftFootSerialPopUpButton.isEnabled = viewModel.leftFootEnabled
        // 有効/無効変更時に接続状態を更新
        viewModel.connectToSerialPorts()
    }
    
    @objc private func refreshSerialPorts(_ sender: NSButton) {
        viewModel.refreshSerialPorts()
        updateSerialPortPopUps()
    }
    
    @objc private func selectSaveLocation(_ sender: NSButton) {
        let openPanel = NSOpenPanel()
        openPanel.canChooseDirectories = true
        openPanel.canChooseFiles = false
        openPanel.allowsMultipleSelection = false
        
        if openPanel.runModal() == .OK {
            if let url = openPanel.url {
                viewModel.saveLocation = url.path
                sender.title = url.lastPathComponent
            }
        }
    }
    
    @objc private func toggleMeasurement(_ sender: NSButton) {
        viewModel.toggleMeasurement()
        sender.title = viewModel.isMeasuring ? "測定停止" : "測定開始"
    }
}

// MARK: - SerialManagerDelegate
extension ControlPanelViewModel: SerialManagerDelegate {
    func serialManager(_ manager: SerialManager, didReceiveData data: String) {
        dataManager?.processSerialData(data)
    }
    
    func serialManager(_ manager: SerialManager, didDisconnectWithError error: Error?) {
        DispatchQueue.main.async {
            self.connectionStatus = "接続エラー"
            if let error = error {
                print("Serial connection error: \(error)")
            }
        }
    }
}

// MARK: - DataManagerDelegate
extension ControlPanelViewModel: DataManagerDelegate {
    func dataManager(_ manager: DataManager, didReceiveFootData data: FootPressureData) {
        DispatchQueue.main.async {
            self.footDataDelegate?.didReceiveFootData(
                rightFoot: data.rightFootData,
                leftFoot: data.leftFootData
            )
        }
    }
}
