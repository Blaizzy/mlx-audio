//
//  SidebarView.swift
//  MLXAudio
//
//  Created by Rudrank Riyam on 6/11/25.
//

import SwiftUI

struct SidebarView: View {
    @Binding var selection: SidebarItem

    var body: some View {
        #if os(macOS)
        List(SidebarItem.allCases, selection: $selection) { item in
            Label(item.rawValue, systemImage: item.icon)
        }
        .listStyle(.sidebar)
        .navigationTitle("MLX Audio")
        #else
        List {
            ForEach(SidebarItem.allCases) { item in
                Button(action: {
                    selection = item
                }) {
                    Label(item.rawValue, systemImage: item.icon)
                }
                .foregroundColor(selection == item ? .accentColor : .primary)
            }
        }
        .listStyle(.sidebar)
        .navigationTitle("MLX Audio")
        #endif
    }
}
