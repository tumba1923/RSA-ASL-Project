from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.lang import Builder
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.uix.button import Button

Builder.load_file('signlanguagetranslator.kv')

class SignLanguageTranslator(BoxLayout):
    def recognize_sign(self):
        # Placeholder for AI recognition logic
        self.ids.phrases_label.text = "Recognized sign: Hello"

    def show_common_phrases(self):
        # Show the popup with common phrases
        phrases = ['Hello', 'Thank you', 'How are you?', 'Goodbye', 'Yes', 'No']
        content = BoxLayout(orientation='vertical', padding=15, spacing=10)

        scroll = ScrollView(size_hint=(1, 1))
        phrase_label = Label(text="\n".join(phrases), font_size=20)
        scroll.add_widget(phrase_label)
        content.add_widget(scroll)

        close_button = Button(text='Close', font_size=16, size_hint_y=None, height=50)
        popup = Popup(title='Common Phrases', content=content, size_hint=(0.7, 0.6))
        close_button.bind(on_press=popup.dismiss)
        content.add_widget(close_button)

        popup.open()

    def clear_phrases(self):
        self.ids.phrases_label.text = ""

class SignLanguageApp(App):
    def build(self):
        return SignLanguageTranslator()

if __name__ == '__main__':
    SignLanguageApp().run()