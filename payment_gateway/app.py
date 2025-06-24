from flask import Flask, render_template, jsonify, request
import stripe

app = Flask(__name__, static_url_path='', static_folder='static', template_folder='templates')

stripe.api_key = 'sk_test_...'  # Replace with your test secret key

YOUR_DOMAIN = 'http://localhost:5000'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/create-checkout-session', methods=['POST'])
def create_checkout_session():
    try:
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            mode='subscription',
            line_items=[{
                'price': 'price_1OyABC...',  # Your actual test Price ID
                'quantity': 1
            }],
            success_url=YOUR_DOMAIN + '/success.html',
            cancel_url=YOUR_DOMAIN + '/cancel.html',
        )
        return jsonify({'url': checkout_session.url})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/webhook', methods=['POST'])
def stripe_webhook():
    payload = request.data
    sig_header = request.headers.get('stripe-signature')
    endpoint_secret = 'whsec_...'  # From Stripe CLI or Dashboard

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, endpoint_secret)

        if event['type'] == 'checkout.session.completed':
            session = event['data']['object']
            print("✅ Checkout complete for customer:", session['customer'])
        
        elif event['type'] == 'invoice.payment_succeeded':
            print("💰 Payment succeeded")

        elif event['type'] == 'customer.subscription.deleted':
            print("⚠️ Subscription canceled")

    except Exception as e:
        return str(e), 400

    return '', 200

if __name__ == '__main__':
    app.run(port=5000)
