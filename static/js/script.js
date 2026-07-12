    const clearButton = document.getElementById("clear");
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    const predictText = document.getElementById("prediction");
    const pollRate = 100

    ctx.fillStyle = "black";
    ctx.fillRect(0,0,canvas.width,canvas.height);

    const pixelData = Array(784).fill(0);

const intervalId = setInterval(() => {
  fetch('/api/data', {
    method: 'POST',
    headers: {
    'Content-Type': 'application/json'
    },
    body: JSON.stringify(pixelData)
    }).then(response => response.json()).then(data => {
    predictText.textContent = data.prediction;
});
}, pollRate);

function checkUp(calc){
    if(calc - 28 > 0){
        return pixelData[calc - 28];
    }
    return 0;
}

function checkDown(calc){
    if(calc + 28 > 0){
       return pixelData[calc + 28];
    }
    return 0;
}

function alphaCalc(calc){
    let num = 0;
    let currentCalc = calc
    if(pixelData[calc] >= 1){
        return 1;
    }
    num += checkUp(currentCalc);
    num += checkDown(currentCalc);
    currentCalc = calc + 1
    if((currentCalc)%27 != 0){
        num += pixelData[currentCalc];
        num += checkUp(currentCalc);
        num += checkDown(currentCalc);
    }
    currentCalc = calc - 1;
    if((currentCalc)%28 != 0){
        num += pixelData[currentCalc];
        num += checkUp(currentCalc);
        num += checkDown(currentCalc);
    }
    return num/8;
}

function checkAlphaUp(calc){
    if(calc - 28 > 0){
        let alpha = alphaCalc(calc - 28);
        return alpha;
    }
    return 0;
}

function checkAlphaDown(calc){
    if(calc + 28 < 784){
        let alpha = alphaCalc(calc + 28);
        return alpha;
    }
    return 0;
}

function checkNeighbors(calc, x, y, num){
    let currentAlpha = 0;
    let currentCalc = calc;
    let alphaArray = Array(8).fill(0)
    let index = 0;
    currentAlpha = checkAlphaUp(currentCalc);
    alphaArray[index] = currentAlpha;
    index += 1;
    currentAlpha = checkAlphaDown(currentCalc);
    alphaArray[index] = currentAlpha;
    index += 1;
    if((currentCalc)%27 != 0){
        currentCalc = calc + 1;
        currentAlpha = alphaCalc(currentCalc);
        alphaArray[index] = currentAlpha;
        index += 1;
        currentAlpha = checkAlphaUp(currentCalc);
        alphaArray[index] = currentAlpha;
        index += 1;
        currentAlpha = checkAlphaDown(currentCalc);
        alphaArray[index] = currentAlpha;
        index += 1;
    }else{
        for(let i = 0; i < 3; i++){
            alphaArray[index] = 0;
            index += 1;
        }
    }
    currentCalc = calc;
    if((currentCalc)%28 != 0){
        currentCalc = calc - 1;
        currentAlpha = alphaCalc(currentCalc);
        alphaArray[index] = currentAlpha;
        index += 1;
        currentAlpha = checkAlphaUp(currentCalc);
        alphaArray[index] = currentAlpha;
        index += 1;
        currentAlpha = checkAlphaDown(currentCalc);
        alphaArray[index] = currentAlpha;
        index += 1;
    }else{
        for(let i = 0; i < 3; i++){
            alphaArray[index] = 0;
            index += 1;
        }
    }
    currentCalc = calc;
    index = 0;
    currentAlpha = alphaArray[index];
    ctx.fillStyle = `rgba(255, 255, 255, ${currentAlpha})`;
    ctx.fillRect(x , y - num, num, num);
    if(currentCalc - 28 > 0){
        pixelData[currentCalc - 28] = currentAlpha
    }
    index += 1;
    currentAlpha = alphaArray[index];
    ctx.fillStyle = `rgba(255, 255, 255, ${currentAlpha})`;
    ctx.fillRect(x , y + num, num, num);
    if(currentCalc + 28 < 784){
        pixelData[currentCalc + 28] = currentAlpha
    }
    currentCalc = calc + 1;
    if((currentCalc)%27 != 0){
        index += 1;
        currentAlpha = alphaArray[index];
        ctx.fillStyle = `rgba(255, 255, 255, ${currentAlpha})`;
        ctx.fillRect(x + num, y, num, num);
        if(currentCalc < 784){
            pixelData[currentCalc] = currentAlpha
        }
        if(currentCalc - 28 > 0){
            index += 1;
            currentAlpha = alphaArray[index];
            ctx.fillStyle = `rgba(255, 255, 255, ${currentAlpha})`;
            ctx.fillRect(x + num, y - num, num, num);
            if(currentCalc - 28 > 0){
                pixelData[currentCalc - 28] = currentAlpha
            }
        }else{
            index += 1;
        }
        if(currentCalc + 28 > 0){
            index += 1;
            currentAlpha = alphaArray[index];
            ctx.fillStyle = `rgba(255, 255, 255, ${currentAlpha})`;
            ctx.fillRect(x + num, y + num, num, num);
            if(currentCalc + 28 < 784){
                pixelData[currentCalc + 28] = currentAlpha
            }
        }else{
            index += 1;
        }
    }else{
        for(let i = 0; i < 3; i++){
            index += 1;
        }
    }
    currentCalc = calc - 1;
    if((currentCalc)%28 != 0){
        index += 1;
        currentAlpha = alphaArray[index];
        ctx.fillStyle = `rgba(255, 255, 255, ${currentAlpha})`;
        ctx.fillRect(x - num, y, num, num);
        if(currentCalc > 0){
            pixelData[currentCalc] = currentAlpha;
        }
        if(currentCalc - 28 > 0){
            index += 1;
            currentAlpha = alphaArray[index];
            ctx.fillStyle = `rgba(255, 255, 255, ${currentAlpha})`;
            ctx.fillRect(x - num, y - num, num, num);
            if(currentCalc - 28 > 0){
                pixelData[currentCalc - 28] = currentAlpha
            }
        }else{
            index += 1;
        }
        if(currentCalc + 28 > 0){
            index += 1;
            currentAlpha = alphaArray[index];
            ctx.fillStyle = `rgba(255, 255, 255, ${currentAlpha})`;
            ctx.fillRect(x - num, y + num, num, num);
            if(currentCalc + 28 < 784){
                pixelData[currentCalc + 28] = currentAlpha
            }
        }else{
            index += 1;
        }
    }else{
        for(let i = 0; i < 3; i++){
            index += 1;
        }
    }
}

function fillPixel(x, y){
    let num = canvas.width/28;
    let floorX = Math.floor(x/num) * num;
    let floorY = Math.floor(y/num) * num;
    ctx.fillStyle = "white";
    ctx.fillRect(floorX, floorY, num, num);
    let calc = Math.floor(y/num) * 28 + Math.floor(x/num);
    if(calc < 784){
        pixelData[calc] = 1;
    }
    checkNeighbors(calc, floorX, floorY, num);
}

clearButton.addEventListener("click", () => {
    ctx.fillStyle = "black";
    ctx.fillRect(0,0,canvas.width,canvas.height);
    pixelData.fill(0);
});

let drawing = false;

canvas.onmousedown = () => drawing = true;
canvas.onmouseup = () => drawing = false;

canvas.onmousemove = e => {

    if(!drawing) return;

    const rect = canvas.getBoundingClientRect();

    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    fillPixel(x, y);
};