//
//  AppDelegate.swift
//  FootViewer
//
//  Created by Tomo Kikuchi on 2025/05/29.
//

import Cocoa

class AppDelegate: NSObject, NSApplicationDelegate {

    private var window: NSWindow?

    func applicationDidFinishLaunching(_ aNotification: Notification) {
        let viewPosition: CGPoint = CGPoint(x: 100, y: 100)  // 画面端から少し離す
        let viewSize: CGSize = CGSize(width: 1920, height: 800)
        
        // ウィンドウの作成
        let contentRect = NSRect(x: viewPosition.x, y: viewPosition.y,
                                width: viewSize.width, height: viewSize.height)
        
        window = NSWindow(
            contentRect: contentRect,
            styleMask: [.titled, .closable, .resizable, .miniaturizable], // .resizableを追加
            backing: .buffered,
            defer: false
        )
        
        window?.title = "FootViewer - 足圧力可視化アプリ"
        
        // 最小サイズを設定
        window?.minSize = NSSize(width: 900, height: 900)
        
        // ViewControllerを作成
        let viewController = ViewController()
        window?.contentViewController = viewController
        
        // ウィンドウを表示
        window?.makeKeyAndOrderFront(nil)
        
        // ウィンドウを画面中央に配置（オプション）
        // window?.center()
        
        // デバッグ用：実際のウィンドウサイズを確認
        print("Window frame: \(window?.frame ?? .zero)")
        print("Content size: \(window?.contentRect(forFrameRect: window?.frame ?? .zero) ?? .zero)")
    }

    func applicationWillTerminate(_ aNotification: Notification) {
        // Insert code here to tear down your application
    }

    func applicationSupportsSecureRestorableState(_ app: NSApplication) -> Bool {
        return true
    }
}
