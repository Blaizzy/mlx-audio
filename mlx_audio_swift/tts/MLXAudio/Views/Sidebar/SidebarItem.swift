//
//  SidebarItem.swift
//  MLXAudio
//
//  Created by Claude Code
//

import Foundation

enum SidebarItem: String, CaseIterable, Identifiable {
    case textToSpeech = "Text to Speech"

    var id: String { rawValue }

    var icon: String {
        switch self {
        case .textToSpeech: return "text.bubble"
        }
    }
}
