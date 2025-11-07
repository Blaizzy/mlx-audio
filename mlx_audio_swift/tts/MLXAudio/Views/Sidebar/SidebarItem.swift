//
//  SidebarItem.swift
//  MLXAudio
//
//  Created by Rudrank Riyam on 6/11/25.
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
