import { Component, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import emailjs from '@emailjs/browser';
import { environment } from '../../../environments/environment';

@Component({
  selector: 'app-contact',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './contact.html',
  styleUrl: './contact.scss',
})
export class Contact {
  isSubmitting = signal(false);
  isSubmitted = signal(false);
  errorMessage = signal<string | null>(null);

  formData = {
    name: '',
    email: '',
    category: 'access',
    message: '',
  };

  private categoryLabels: Record<string, string> = {
    access: 'Document Access / Permissions',
    indexing: 'Knowledge Base Indexing Request',
    bug: 'Report an Answer Discrepancy',
    general: 'General Inquiry',
  };

  onSubmit(): void {
    if (this.isSubmitting()) return;

    this.isSubmitting.set(true);
    this.errorMessage.set(null);

    const templateParams = {
      from_name: this.formData.name,
      from_email: this.formData.email,
      category: this.categoryLabels[this.formData.category] ?? this.formData.category,
      message: this.formData.message,
      // reply_to is set so the support team can reply directly to the user
      reply_to: this.formData.email,
    };

    const { publicKey, serviceId, templateId, autoReplyTemplateId } = environment.emailjs;

    const emailPromises = [
      emailjs.send(serviceId, templateId, templateParams, { publicKey })
    ];

    // If an auto-reply template is configured, also send confirmation to user
    if (autoReplyTemplateId && !autoReplyTemplateId.includes('YOUR_')) {
      emailPromises.push(
        emailjs.send(serviceId, autoReplyTemplateId, templateParams, { publicKey })
      );
    }

    Promise.all(emailPromises)
      .then(() => {
        this.isSubmitted.set(true);
        this.isSubmitting.set(false);
        this.formData = { name: '', email: '', category: 'access', message: '' };
      })
      .catch((err) => {
        console.error('EmailJS error:', err);
        this.errorMessage.set('Failed to send your message. Please try again or email us directly.');
        this.isSubmitting.set(false);
      });
  }

  sendAnother(): void {
    this.isSubmitted.set(false);
    this.errorMessage.set(null);
  }
}