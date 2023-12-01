const msgerForm = document.querySelector(".msger-inputarea");
const msgerInput = document.querySelector(".msger-input");
const msgerChat = document.querySelector(".msger-chat");
const msgerSendButton = document.querySelector(".msger-send-btn");
const BOT_NAME = "Smart Glass Country";
const PERSON_NAME = "You";

window.addEventListener('load',()=>{
  firstPrompt();
})


async function firstPrompt() {
  if (!sessionStorage.getItem("session_id")) {
    await fetch(`/seeding_sales_conversation`).then((response) => {
      response.json().then((res) => {
        sessionStorage.setItem("session_id", res);
      });
    });
  }
  text = "Hello, I am David from Smart Glass Country.  ";
  appendMessage(BOT_NAME, "left", text);
}

// Function to print messages on display
function appendMessage(name, side, text) {
  const msgHTML = `
    <div class="msg ${side}-msg">
      
      <div class="msg-bubble">
        <div class="msg-info">
          <div class="msg-info-name">${name}</div>
        </div>
        <div class="msg-text">${text}</div>
      </div>
    </div>
  `;
  msgerChat.insertAdjacentHTML("beforeend", msgHTML);
  msgerChat.scrollTop += 1000;
}

// Form submit event
msgerForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const msgText = msgerInput.value;
  if (!msgText) return;

  appendMessage(PERSON_NAME, "right", msgText);
  msgerInput.value = "";
  await botResponse(msgText);
});

async function botResponse(query) {
  toggleTyping(true);
  msgerSendButton.disabled = true;
  await fetch(`/sales_conversation?query=${query}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Session-Id": sessionStorage.getItem("session_id"),
    },
  })
    .then((response) => {
      console.log(response);
      if (response.status < 200 || response.status >= 300) {
        let error = new Error(response.statusText);
        throw error;
      }
      response.json().then((res) => {
        const msg = res;
        appendMessage(BOT_NAME, "left", msg);
        msgerSendButton.disabled = false;
        toggleTyping(false);
      });
    })
    .catch((error) => {
      // const msg = error;
      const msg = "Something went wrong! Please try again later.";
      const msgHTML = `<p>Oops, something went wrong on our end!</p>
      <br>
      <p>
        We're sorry, but it looks like there's a temporary issue with our system.
        Our team is already on the case and working to fix it as quickly as possible.
        We understand how important this is for you, and we appreciate your patience.
      </p>
      <br>
      <p>
        In the meantime, here are a couple of things you can try:
      </p>
      <ul style="margin-left: 20px;">
        <li>Check back in a little while and things should be up and running again.</li>
        <li>If the problem persists, please don't hesitate to reach out to our support team at 1-800-791-1977 for further assistance.</li>
      </ul>
      <br>
      <p>Thank you for your understanding and cooperation.</p>
      `;
      appendMessage(BOT_NAME, "left", msgHTML);
      msgerSendButton.disabled = true;
      toggleTyping(false);
    });
}

function toggleTyping(status) {
  var typingStatus = document.getElementById("typing");
  if (status) {
    typingStatus.style.display = "block";
  } else {
    typingStatus.style.display = "none";
  }
}
