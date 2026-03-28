# kinnect.py
"""
KINNECT AI - Main Entry Point

Unified interface for both text and voice conversations.
100% REAL - No simulation, no hardcoded responses.
"""

import sys
import os

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   ██╗  ██╗██╗███╗   ██╗███╗   ██╗███████╗ ██████╗████████╗           ║
║   ██║ ██╔╝██║████╗  ██║████╗  ██║██╔════╝██╔════╝╚══██╔══╝           ║
║   █████╔╝ ██║██╔██╗ ██║██╔██╗ ██║█████╗  ██║        ██║              ║
║   ██╔═██╗ ██║██║╚██╗██║██║╚██╗██║██╔══╝  ██║        ██║              ║
║   ██║  ██╗██║██║ ╚████║██║ ╚████║███████╗╚██████╗   ██║              ║
║   ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═══╝╚══════╝ ╚═════╝   ╚═╝              ║
║                                                                      ║
║            AI-Powered Cognitive Health Companion                     ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

def main():
    print_banner()
    
    print("Welcome to Kinnect AI - Your Cognitive Health Companion\n")
    print("Choose conversation mode:")
    print("  1. Voice Chat (Microphone + Speaker)")
    print("  2. Text Chat (Keyboard + Screen)")
    print("  3. Exit\n")
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice == "1":
        # Voice mode
        user_id = input("\nEnter patient ID (or press Enter for 'patient_001'): ").strip()
        if not user_id:
            user_id = "patient_001"
        
        print("\nStarting voice conversation...")
        print("Make sure your microphone and speakers are ready.\n")
        
        from backend.voice_chat import KinnectVoiceChat
        
        chat = KinnectVoiceChat(
            user_id=user_id,
            whisper_model="base",
            tts_method="gtts"
        )
        
        try:
            chat.run_conversation()
        except KeyboardInterrupt:
            print("\n\n⚠️ Conversation interrupted")
        
    elif choice == "2":
        # Text mode
        user_id = input("\nEnter patient ID (or press Enter for 'patient_001'): ").strip()
        if not user_id:
            user_id = "patient_001"
        
        print("\nStarting text conversation...")
        
        from backend.cli_chat import KinnectCLI
        
        cli = KinnectCLI(user_id=user_id)
        cli.run_interactive_session()
        
    elif choice == "3":
        print("\nGoodbye!\n")
        sys.exit(0)
    else:
        print("\n❌ Invalid choice. Please enter 1, 2, or 3.\n")
        main()

if __name__ == "__main__":
    main()