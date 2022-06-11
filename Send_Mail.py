# Send mails form your gmail account
import smtplib
# creates SMTP connection
_s_ = smtplib.SMTP('smtp.gmail.com', 587)
# starts TLS for Security
_s_.starttls()
# Authenticates
_s_.login("sender_mail", "sender_email_password",)
# Message to be sent
_message_ = "Enter your message here"
# Sending mail
_s_.sendmail("sender_mail", "recevier_mail", _message_)
# terminate
_s_.quit()