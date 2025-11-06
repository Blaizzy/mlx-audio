//
//  SidebarView.swift
//  Swift-TTS
//
//  Created by Claude Code
//

import SwiftUI

struct SidebarView: View {
    @Binding var selection: SidebarItem

    var body: some View {
        List(SidebarItem.allCases, selection: $selection) { item in
            Label(item.rawValue, systemImage: item.icon)
        }
        .listStyle(.sidebar)
        .navigationTitle("MLX Audio")
    }
}
