//
//  ViewController.swift
//  FootViewer
//
//  Created by Tomo Kikuchi on 2025/05/29.
//

import Cocoa

class ViewController: NSViewController {

    private lazy var controlPanelView: ControlPanelView = generateControlPanelView()
    private lazy var rightFootView: FootView = generateRightFootView()
    private lazy var leftFootView: FootView = generateLeftFootView()
    
    // 足のビューを含むコンテナ
    private lazy var footContainerView: NSView = generateFootContainerView()

    override func viewDidLoad() {
        super.viewDidLoad()
        setup()
    }
    
    override func viewWillAppear() {
        super.viewWillAppear()
        
        // ビューが表示される前にサイズを確認・調整
        if let window = view.window {
            print("ViewWillAppear - Window frame: \(window.frame)")
            print("ViewWillAppear - Content size: \(window.contentRect(forFrameRect: window.frame))")
        }
    }
    
    override func viewDidAppear() {
        super.viewDidAppear()
        
        // ビューが表示された後の最終調整
        if let window = view.window {
            print("ViewDidAppear - Window frame: \(window.frame)")
            print("ViewDidAppear - View bounds: \(view.bounds)")
        }
    }

    override var representedObject: Any? {
        didSet {
            // Update the view, if already loaded.
        }
    }

    func setup() {
        layoutControlPanelView()
        layoutFootContainerView()
        layoutFootViews()
        
        // ControlPanelViewModelのデリゲートを設定
        controlPanelView.viewModel.footDataDelegate = self
        
        // テスト用のダミーデータを設定
        setupTestData()
    }
    
    // MARK: - Test Data
    private func setupTestData() {
        // 右足用のテストデータ
        let rightFootTestData: [Double] = [
            0.2, 0.4, 0.6, 0.3,  // つま先
            0.8, 0.7, 0.5, 0.9,  // 前足部
            0.4, 0.6, 0.7,       // 中足部
            0.3, 0.5, 0.4,       // 後足部
            0.6, 0.8             // かかと
        ]
        
        // 左足用のテストデータ（少し異なる値）
        let leftFootTestData: [Double] = [
            0.3, 0.5, 0.7, 0.4,  // つま先
            0.9, 0.6, 0.4, 0.8,  // 前足部
            0.5, 0.7, 0.6,       // 中足部
            0.2, 0.4, 0.5,       // 後足部
            0.7, 0.9             // かかと
        ]
        
        rightFootView.updatePressureData(rightFootTestData)
        leftFootView.updatePressureData(leftFootTestData)
    }
    
    // MARK: - Public Methods
    
    /// 右足の圧力データを更新
    public func updateRightFootData(_ data: [Double]) {
        rightFootView.updatePressureData(data)
    }
    
    /// 左足の圧力データを更新
    public func updateLeftFootData(_ data: [Double]) {
        leftFootView.updatePressureData(data)
    }
    
    /// 両足の圧力データを同時に更新
    public func updateBothFeetData(rightFoot: [Double], leftFoot: [Double]) {
        rightFootView.updatePressureData(rightFoot)
        leftFootView.updatePressureData(leftFoot)
    }
    
    /// 両足のデータをリセット
    public func resetBothFeetData() {
        rightFootView.resetPressureData()
        leftFootView.resetPressureData()
    }
}

extension ViewController {
    func generateControlPanelView() -> ControlPanelView {
        let view = ControlPanelView()
        view.translatesAutoresizingMaskIntoConstraints = false
        return view
    }

    func generateRightFootView() -> FootView {
        let view = FootView(frame: .zero, footType: .right)
        view.translatesAutoresizingMaskIntoConstraints = false
        return view
    }
    
    func generateLeftFootView() -> FootView {
        let view = FootView(frame: .zero, footType: .left)
        view.translatesAutoresizingMaskIntoConstraints = false
        return view
    }
    
    func generateFootContainerView() -> NSView {
        let view = NSView()
        view.translatesAutoresizingMaskIntoConstraints = false
        view.wantsLayer = true
        view.layer?.backgroundColor = NSColor.controlBackgroundColor.cgColor
        return view
    }

    func layoutControlPanelView() {
        view.addSubview(controlPanelView)

        NSLayoutConstraint.activate([
            controlPanelView.topAnchor.constraint(equalTo: view.topAnchor),
            controlPanelView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            controlPanelView.leftAnchor.constraint(equalTo: view.leftAnchor),
            controlPanelView.widthAnchor.constraint(equalToConstant: 300),
        ])
    }

    func layoutFootContainerView() {
        view.addSubview(footContainerView)

        NSLayoutConstraint.activate([
            footContainerView.rightAnchor.constraint(equalTo: view.rightAnchor),
            footContainerView.leftAnchor.constraint(equalTo: controlPanelView.rightAnchor),
            footContainerView.topAnchor.constraint(equalTo: view.topAnchor),
            footContainerView.bottomAnchor.constraint(equalTo: view.bottomAnchor)
        ])
    }
    
    func layoutFootViews() {
        // 右足ビューを追加
        footContainerView.addSubview(rightFootView)
        footContainerView.addSubview(leftFootView)
        
        // 足のラベルを追加
        let rightFootLabel = createFootLabel(text: "右足 (Right)")
        let leftFootLabel = createFootLabel(text: "左足 (Left)")
        
        footContainerView.addSubview(rightFootLabel)
        footContainerView.addSubview(leftFootLabel)
        
        NSLayoutConstraint.activate([
            // 左足ラベル（左側に配置）
            leftFootLabel.topAnchor.constraint(equalTo: footContainerView.topAnchor, constant: 10),
            leftFootLabel.centerXAnchor.constraint(equalTo: leftFootView.centerXAnchor),
            
            // 左足ビュー（左側に配置）
            leftFootView.topAnchor.constraint(equalTo: leftFootLabel.bottomAnchor, constant: 5),
            leftFootView.leftAnchor.constraint(equalTo: footContainerView.leftAnchor, constant: 10),
            leftFootView.bottomAnchor.constraint(equalTo: footContainerView.bottomAnchor, constant: -10),
            leftFootView.widthAnchor.constraint(equalTo: footContainerView.widthAnchor, multiplier: 0.48),
            
            // 右足ラベル（右側に配置）
            rightFootLabel.topAnchor.constraint(equalTo: footContainerView.topAnchor, constant: 10),
            rightFootLabel.centerXAnchor.constraint(equalTo: rightFootView.centerXAnchor),
            
            // 右足ビュー（右側に配置）
            rightFootView.topAnchor.constraint(equalTo: rightFootLabel.bottomAnchor, constant: 5),
            rightFootView.rightAnchor.constraint(equalTo: footContainerView.rightAnchor, constant: -10),
            rightFootView.bottomAnchor.constraint(equalTo: footContainerView.bottomAnchor, constant: -10),
            rightFootView.widthAnchor.constraint(equalTo: footContainerView.widthAnchor, multiplier: 0.48),
            
            // 左右の足ビュー間にスペースを確保
            rightFootView.leftAnchor.constraint(equalTo: leftFootView.rightAnchor, constant: 10)
        ])
    }
    
    private func createFootLabel(text: String) -> NSTextField {
        let label = NSTextField()
        label.isEditable = false
        label.isBordered = false
        label.isSelectable = false
        label.stringValue = text
        label.font = NSFont.boldSystemFont(ofSize: 14)
        label.textColor = NSColor.labelColor
        label.backgroundColor = NSColor.clear
        label.drawsBackground = false
        label.alignment = .center
        label.translatesAutoresizingMaskIntoConstraints = false
        return label
    }
}

// MARK: - FootDataDelegate
extension ViewController: FootDataDelegate {
    func didReceiveFootData(rightFoot: [Double], leftFoot: [Double]) {
        DispatchQueue.main.async {
            self.rightFootView.updatePressureData(rightFoot)
            self.leftFootView.updatePressureData(leftFoot)
        }
    }
}
