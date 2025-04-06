<template>
<body class="w-screen h-screen">
 <div :class="{'blur-sm':uploadPopup}" class=" px-5 py-5 w-screen h-screen gap-x-2 flex flex-row">
 
 <div class="bg-slate-100 bg-opacity-100 h-full w-[18rem] rounded-lg border-[0.5px] border-slate-200 flex flex-col pt-4 px-4">
 
 <div class="flex flex-row gap-x-3 ">
 
 <img class="h-8 w-8 bg-black my-auto rounded-full" src="https://static.vecteezy.com/system/resources/thumbnails/009/292/244/small/default-avatar-icon-of-social-media-user-vector.jpg">
 <p class="my-auto">John Doe</p>
 </div>
 
 
 <hr class="h-px my-4 bg-slate-300 border-0">
 
 <button @click="uploadPopup = true" class="flex flex-row bg-gradient-to-b to-[#17183B] from-[#A11692] rounded-lg transition duration-200 hover:opacity-85 h-[2.75rem] transition duration-200 hover:scale-[1.01] w-full ">
 
 <div class="gap-x-2 mx-auto my-auto w-full h-full justify-center flex flex-row group">
 <svg xmlns="http://www.w3.org/2000/svg" class="my-auto stroke-white stroke-[60] w-4 h-4 transition duration-200 group-hover:rotate-[20deg]" viewBox="0 0 512 512"><path fill="none" stroke="white" stroke-linecap="round" stroke-linejoin="round" d="M256 112v288M400 256H112"/></svg>
 <p class="text-white my-auto text-sm">Upload Memories</p>
 </div>
 </button>
 
 <hr class="h-px my-4 bg-slate-300 border-0">
 
 <div class="flex flex-col relative h-full">
 <div class="flex flex-col gap-y-1">
 <div class="w-full h-full" v-for="(item,index) in vNav" :key="item.name" >
 <button @click="itemSelected = item.name" :class="{ 'bg-slate-200': itemSelected==item.name }" class="group rounded-lg text-sm py-[0.5rem] px-3 flex flex-row gap-x-2 w-full transition duration-200 hover:scale-[1.02] hover:bg-slate-200">
 <svg xmlns="http://www.w3.org/2000/svg" v-if="item.name=='My Storage'" class="w-4 h-5 my-auto transition duration-200 group-hover:rotate-[20deg]" viewBox="0 0 512 512"><path d="M408 96H252.11a23.89 23.89 0 01-13.31-4L211 73.41A55.77 55.77 0 00179.89 64H104a56.06 56.06 0 00-56 56v24h416c0-30.88-25.12-48-56-48zM423.75 448H88.25a56 56 0 01-55.93-55.15L16.18 228.11v-.28A48 48 0 0164 176h384.1a48 48 0 0147.8 51.83v.28l-16.22 164.74A56 56 0 01423.75 448zm56.15-221.45z"/></svg>
 <svg xmlns="http://www.w3.org/2000/svg" v-if="item.name=='Shared with me'" class="w-4 h-5 my-auto transition duration-200 group-hover:rotate-[-20deg]" viewBox="0 0 512 512"><path d="M340.75 344.49c5.91-20.7 9.82-44.75 11.31-67.84a4.41 4.41 0 00-4.46-4.65h-71.06a4.43 4.43 0 00-4.47 4.39v55.3a4.44 4.44 0 004.14 4.38 273.51 273.51 0 0159 11.39 4.45 4.45 0 005.54-2.97zM323.58 377.31a260.05 260.05 0 00-46.6-9.09 4.42 4.42 0 00-4.91 4.29v65.24a4.47 4.47 0 006.76 3.7c15.9-9.27 29-24.84 40.84-45.43 1.94-3.36 4.89-9.15 6.67-12.69a4.29 4.29 0 00-2.76-6.02zM235.29 368.4a256.85 256.85 0 00-46.56 8.82c-2.64.76-3.75 4.4-2.55 6.79 1.79 3.56 4 8.11 5.89 11.51 13 23 26.84 37.5 41.24 45.93a4.47 4.47 0 006.76-3.7v-65.27a4.16 4.16 0 00-4.78-4.08zM235.6 272h-71.06a4.41 4.41 0 00-4.46 4.64c1.48 23.06 5.37 47.16 11.26 67.84a4.46 4.46 0 005.59 3 272.2 272.2 0 0159-11.36 4.44 4.44 0 004.15-4.38V276.4a4.43 4.43 0 00-4.48-4.4zM277 143.78a235.8 235.8 0 0046.5-9.14 4.3 4.3 0 002.76-6c-1.79-3.57-4.27-8.68-6.17-12.09-12.29-22-26.14-37.35-41.24-46a4.48 4.48 0 00-6.76 3.7v65.23a4.43 4.43 0 004.91 4.3zM276.54 240h71.06a4.39 4.39 0 004.46-4.58c-1.48-22.77-5.27-47.8-11.16-68.22a4.46 4.46 0 00-5.59-2.95c-19 5.74-38.79 10.43-59.09 12a4.4 4.4 0 00-4.15 4.32v55.11a4.4 4.4 0 004.47 4.32zM233.31 70.56c-15.42 8.57-29.17 24.43-41.47 46.37-1.91 3.41-4.19 8.11-6 11.67a4.31 4.31 0 002.76 6 225.42 225.42 0 0046.54 9.17 4.43 4.43 0 004.91-4.29V74.26a4.49 4.49 0 00-6.74-3.7zM235.92 176.26c-20.3-1.55-40.11-6.24-59.09-12a4.46 4.46 0 00-5.59 2.95c-5.89 20.42-9.68 45.45-11.16 68.22a4.39 4.39 0 004.46 4.58h71.06a4.4 4.4 0 004.47-4.34v-55.09a4.4 4.4 0 00-4.15-4.32z"/><path d="M414.39 97.61A224 224 0 1097.61 414.39 224 224 0 10414.39 97.61zM176.6 430.85a219.08 219.08 0 01-12.48-19.66c-2-3.69-4.84-9.26-6.73-13.13a7.29 7.29 0 00-10.31-3.16c-4.3 2.41-10 5.72-14.13 8.43a147.29 147.29 0 01-23.57-22.43 248.83 248.83 0 0130.41-18.36c1.86-1 2.77-2.14 2.18-4.18a374.8 374.8 0 01-14.09-82.17 4.36 4.36 0 00-4.3-4.17H66.84a2 2 0 01-2-1.7A98.28 98.28 0 0164 256a96.27 96.27 0 01.86-14.29 2 2 0 012-1.7h56.74c2.29 0 4.17-1.32 4.29-3.63a372.71 372.71 0 0114-81.83 4.36 4.36 0 00-2.19-5.11 260.63 260.63 0 01-29.84-17.9 169.82 169.82 0 0123.14-22.8c4.08 2.68 9.4 5.71 13.66 8.11a7.89 7.89 0 0011-3.42c1.88-3.87 4-8.18 6.06-11.88a221.93 221.93 0 0112.54-19.91A185 185 0 01256 64c28.94 0 55.9 7 80.53 18.46a202.23 202.23 0 0112 19c2.59 4.66 5.34 10.37 7.66 15.32a4.29 4.29 0 005.92 1.94c5.38-2.91 11.21-6.26 16.34-9.63a171.36 171.36 0 0123.2 23 244.89 244.89 0 01-29.06 17.31 4.35 4.35 0 00-2.18 5.12 348.68 348.68 0 0113.85 81.4 4.33 4.33 0 004.3 4.12l56.62-.07a2 2 0 012 1.7 117.46 117.46 0 010 28.62 2 2 0 01-2 1.72h-56.67a4.35 4.35 0 00-4.3 4.17 367.4 367.4 0 01-13.87 81.3 4.45 4.45 0 002.19 5.19c5 2.59 10.57 5.48 15.37 8.42s9.55 6.08 14.13 9.34a172.73 172.73 0 01-23 22.93c-2.44-1.61-5.34-3.44-7.84-4.94-1.72-1-4.89-2.77-6.65-3.76-3.82-2.14-7.88-.54-9.79 3.4s-4.83 9.59-6.87 13.25a212.42 212.42 0 01-12.35 19.53C310.91 442.37 284.94 448 256 448s-54.77-5.63-79.4-17.15z"/></svg>
 <svg xmlns="http://www.w3.org/2000/svg" v-if="item.name =='Recent Files'" class="w-4 h-5 my-auto transition duration-200 group-hover:rotate-[20deg]" viewBox="0 0 512 512"><path d="M256 48C141.13 48 48 141.13 48 256s93.13 208 208 208 208-93.13 208-208S370.87 48 256 48zm96 240h-96a16 16 0 01-16-16V128a16 16 0 0132 0v128h80a16 16 0 010 32z"/></svg>
 <svg xmlns="http://www.w3.org/2000/svg" v-if="item.name =='Your Favorites'" class="w-4 h-5 my-auto transition duration-200 group-hover:rotate-[-20deg]" viewBox="0 0 512 512"><path d="M394 480a16 16 0 01-9.39-3L256 383.76 127.39 477a16 16 0 01-24.55-18.08L153 310.35 23 221.2a16 16 0 019-29.2h160.38l48.4-148.95a16 16 0 0130.44 0l48.4 149H480a16 16 0 019.05 29.2L359 310.35l50.13 148.53A16 16 0 01394 480z"/></svg>
 <svg xmlns="http://www.w3.org/2000/svg" v-if="item.name =='Saved Chats'" class="w-4 h-5 my-auto transition duration-200 group-hover:rotate-[-20deg]" viewBox="0 0 512 512"><path d="M400 480a16 16 0 01-10.63-4L256 357.41 122.63 476A16 16 0 0196 464V96a64.07 64.07 0 0164-64h192a64.07 64.07 0 0164 64v368a16 16 0 01-16 16z"/></svg>
 {{ item.name }}</button>
 </div>
 
 <hr class="h-px my-2 bg-slate-300 border-0">
 
 
 </div>
 
 
 
 
 
 
 <div class="flex flex-col gap-y-1 absolute bottom-4 w-full">
 
 <div class="bg-slate-200 w-full h-[10rem] rounded-lg py-4 px-4 flex flex-col gap-y-4 transition duration-200 hover:scale-[1.01] group">
 <div class="flex flex-row gap-x-2 ">
 <div class="bg-white py-1 px-2 rounded-lg my-auto">
 <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-5 my-auto transition duration-200 group-hover:rotate-[-90deg]" viewBox="0 0 512 512"><path d="M440.9 136.3a4 4 0 000-6.91L288.16 40.65a64.14 64.14 0 00-64.33 0L71.12 129.39a4 4 0 000 6.91L254 243.88a4 4 0 004.06 0zM54 163.51a4 4 0 00-6 3.49v173.89a48 48 0 0023.84 41.39L234 479.51a4 4 0 006-3.46V274.3a4 4 0 00-2-3.46zM272 275v201a4 4 0 006 3.46l162.15-97.23A48 48 0 00464 340.89V167a4 4 0 00-6-3.45l-184 108a4 4 0 00-2 3.45z"/></svg>
 </div>
 <p class="text-sm my-auto font-bold">Your Usage</p> 
 </div>



 <div class="flex flex-col w-full gap-y-4">

 <div class="flex flex-col gap-y-1">
 <div class="text-xs flex flex-row justify-between">
 <p>Cloud Storage</p> 
 <p>10/30 GB</p></div>
 <div class="w-full rounded-full h-2.5 bg-[#17183B]">
 <div class="bg-gradient-to-l to-[#17183B] from-[#A11692] h-2.5 rounded-full" style="width: 33%"></div>
 </div>
 </div>

 <div class="flex flex-col gap-y-1">
 <div class="text-xs flex flex-row justify-between">
 <p>Vector Searches</p> 
 <p>8/10</p></div>
 <div class="w-full rounded-full h-2.5 bg-[#17183B]">
 <div class="bg-gradient-to-l to-[#17183B] from-[#A11692] h-2.5 rounded-full" style="width: 80%"></div>
 </div>
 </div>

 
 </div>
 
 </div>
 <hr class="h-px my-2 bg-slate-300 border-0">
 
 <button @click="itemSelected = 'Settings'" :class="{ 'bg-slate-200': itemSelected=='Settings' }" class="group rounded-lg text-sm py-[0.5rem] px-3 flex flex-row gap-x-2 w-full transition duration-200 hover:scale-[1.02] hover:bg-slate-200">
 <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-5 my-auto transition duration-200 group-hover:rotate-[-20deg]" viewBox="0 0 512 512"><circle cx="256" cy="256" r="48"/><path d="M470.39 300l-.47-.38-31.56-24.75a16.11 16.11 0 01-6.1-13.33v-11.56a16 16 0 016.11-13.22L469.92 212l.47-.38a26.68 26.68 0 005.9-34.06l-42.71-73.9a1.59 1.59 0 01-.13-.22A26.86 26.86 0 00401 92.14l-.35.13-37.1 14.93a15.94 15.94 0 01-14.47-1.29q-4.92-3.1-10-5.86a15.94 15.94 0 01-8.19-11.82l-5.59-39.59-.12-.72A27.22 27.22 0 00298.76 26h-85.52a26.92 26.92 0 00-26.45 22.39l-.09.56-5.57 39.67a16 16 0 01-8.13 11.82 175.21 175.21 0 00-10 5.82 15.92 15.92 0 01-14.43 1.27l-37.13-15-.35-.14a26.87 26.87 0 00-32.48 11.34l-.13.22-42.77 73.95a26.71 26.71 0 005.9 34.1l.47.38 31.56 24.75a16.11 16.11 0 016.1 13.33v11.56a16 16 0 01-6.11 13.22L42.08 300l-.47.38a26.68 26.68 0 00-5.9 34.06l42.71 73.9a1.59 1.59 0 01.13.22 26.86 26.86 0 0032.45 11.3l.35-.13 37.07-14.93a15.94 15.94 0 0114.47 1.29q4.92 3.11 10 5.86a15.94 15.94 0 018.19 11.82l5.56 39.59.12.72A27.22 27.22 0 00213.24 486h85.52a26.92 26.92 0 0026.45-22.39l.09-.56 5.57-39.67a16 16 0 018.18-11.82c3.42-1.84 6.76-3.79 10-5.82a15.92 15.92 0 0114.43-1.27l37.13 14.95.35.14a26.85 26.85 0 0032.48-11.34 2.53 2.53 0 01.13-.22l42.71-73.89a26.7 26.7 0 00-5.89-34.11zm-134.48-40.24a80 80 0 11-83.66-83.67 80.21 80.21 0 0183.66 83.67z"/></svg> Settings</button>
 
 <button class="group rounded-lg hover:bg-slate-200 text-sm py-[0.5rem] px-3 flex flex-row gap-x-2 w-full transition duration-200 hover:scale-[1.02] hover:bg-slate-200">
 <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-5 my-auto transition duration-200 group-hover:rotate-[20deg]" viewBox="0 0 512 512"><path d="M160 256a16 16 0 0116-16h144V136c0-32-33.79-56-64-56H104a56.06 56.06 0 00-56 56v240a56.06 56.06 0 0056 56h160a56.06 56.06 0 0056-56V272H176a16 16 0 01-16-16zM459.31 244.69l-80-80a16 16 0 00-22.62 22.62L409.37 240H320v32h89.37l-52.68 52.69a16 16 0 1022.62 22.62l80-80a16 16 0 000-22.62z"/></svg> 
 Sign Out</button>
 </div>
 </div>
 
 
 
 </div>

 <div class="w-full h-full rounded-lg flex flex-row gap-x-2">

 
 <div v-if="itemSelected=='My Storage'" class=" bg-slate-100 rounded-lg h-full w-[65%] px-10 py-10 flex flex-col">

 <div class="flex flex-col gap-y-2">
 <div class="font-bold text-xl">All Files</div>

 <div class="flex flex-row bg-slate-200 w-full rounded-md py-3 text-slate-600 px-6 text-sm">
 <p class="my-auto line-clamp-[1]">Name</p>
 <p class="my-auto ml-[12rem] line-clamp-[1]">Size</p>
 <p class="my-auto ml-[5.6rem] line-clamp-[1]">Last Modified</p>

 </div>

<div class="overflow-y-scroll h-full mt-2 pb-2">
 <div v-for="(item, index) in allFiles" class="border-b-[1.5px] rounded-lg mb-1 border-slate-200">
 <div class="flex flex-row text-xs gap-x-3 bg-white py-[0.75rem] px-5 transition duration-200 hover:bg-slate-200 rounded-lg ">

 <div class="group rounded-lg text-sm py-2 flex flex-row gap-x-1 w-fit px-2 rounded-md bg-slate-300">
 <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4 my-auto fill-slate-400" viewBox="0 0 512 512"><path d="M428 224H288a48 48 0 01-48-48V36a4 4 0 00-4-4h-92a64 64 0 00-64 64v320a64 64 0 0064 64h224a64 64 0 0064-64V228a4 4 0 00-4-4z"/><path d="M419.22 188.59L275.41 44.78a2 2 0 00-3.41 1.41V176a16 16 0 0016 16h129.81a2 2 0 001.41-3.41z"/></svg> 



 </div>

 <p class="my-auto w-[15rem] line-clamp-[1]">{{item.metaData.name }}</p>
 <p class="my-auto ml-6 w-[8rem] line-clamp-[1]">{{ (item.metaData.size / 1000000).toFixed(2) }} MB</p>
 <p class="my-auto ml-6 w-[20rem] line-clamp-[1]">
 {{ new Date(item.metaData.timeCreated).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' }) }}
</p>


<button class="group rounded-lg text-sm flex flex-row gap-x-1 w-fit rounded-md" @click="deleteFile(item.metaData.name)">
 <svg xmlns="http://www.w3.org/2000/svg" class="w-5 mt-2 h-5 my-auto fill-slate-400 transition duration-200 hover:fill-black" viewBox="0 0 512 512"><path d="M296 64h-80a7.91 7.91 0 00-8 8v24h96V72a7.91 7.91 0 00-8-8z" fill="none"/><path d="M432 96h-96V72a40 40 0 00-40-40h-80a40 40 0 00-40 40v24H80a16 16 0 000 32h17l19 304.92c1.42 26.85 22 47.08 48 47.08h184c26.13 0 46.3-19.78 48-47l19-305h17a16 16 0 000-32zM192.57 416H192a16 16 0 01-16-15.43l-8-224a16 16 0 1132-1.14l8 224A16 16 0 01192.57 416zM272 400a16 16 0 01-32 0V176a16 16 0 0132 0zm32-304h-96V72a7.91 7.91 0 018-8h80a7.91 7.91 0 018 8zm32 304.57A16 16 0 01320 416h-.58A16 16 0 01304 399.43l8-224a16 16 0 1132 1.14z"/></svg>
</button>




 <a class="group rounded-lg text-sm flex flex-row gap-x-1 w-fit rounded-md" :href="item.URL" :download="item.metaData.name">
 <svg xmlns="http://www.w3.org/2000/svg" class="w-6 h-6 -ml-2 my-auto fill-slate-400 transition duration-200 hover:fill-black" viewBox="0 0 512 512"><path d="M376 160H272v153.37l52.69-52.68a16 16 0 0122.62 22.62l-80 80a16 16 0 01-22.62 0l-80-80a16 16 0 0122.62-22.62L240 313.37V160H136a56.06 56.06 0 00-56 56v208a56.06 56.06 0 0056 56h240a56.06 56.06 0 0056-56V216a56.06 56.06 0 00-56-56zM272 48a16 16 0 00-32 0v112h32z"/></svg>
 </a>



 </div>
 </div>
 </div>
 
 </div>


 
 </div> 










 <div v-if="itemSelected=='Deleted Files'" class=" bg-slate-100 rounded-lg h-full w-[65%] px-10 py-10 flex flex-col">




</div> 
















 


 <div class="bg-slate-100 rounded-lg h-full w-[35%] max-w-[25rem] py-3 px-3 border-[0.5px] border-slate-200 ">
 <div class="relative h-full w-full flex flex-col gap-y-2">
 <div class="flex flex-row gap-x-2">
 <svg xmlns="http://www.w3.org/2000/svg" class="w-5 h-5 my-auto fill-slate-400" viewBox="0 0 512 512"><path d="M60.44 389.17c0 .07 0 .2-.08.38.03-.12.05-.25.08-.38zM439.9 405.6a26.77 26.77 0 01-9.59-2l-56.78-20.13-.42-.17a9.88 9.88 0 00-3.91-.76 10.32 10.32 0 00-3.62.66c-1.38.52-13.81 5.19-26.85 8.77-7.07 1.94-31.68 8.27-51.43 8.27-50.48 0-97.68-19.4-132.89-54.63A183.38 183.38 0 01100.3 215.1a175.9 175.9 0 014.06-37.58c8.79-40.62 32.07-77.57 65.55-104A194.76 194.76 0 01290.3 32c52.21 0 100.86 20 137 56.18 34.16 34.27 52.88 79.33 52.73 126.87a177.86 177.86 0 01-30.3 99.15l-.19.28-.74 1c-.17.23-.34.45-.5.68l-.15.27a21.63 21.63 0 00-1.08 2.09l15.74 55.94a26.42 26.42 0 011.12 7.11 24 24 0 01-24.03 24.03z"/><path d="M299.87 425.39a15.74 15.74 0 00-10.29-8.1c-5.78-1.53-12.52-1.27-17.67-1.65a201.78 201.78 0 01-128.82-58.75A199.21 199.21 0 0186.4 244.16C85 234.42 85 232 85 232a16 16 0 00-28-10.58s-7.88 8.58-11.6 17.19a162.09 162.09 0 0011 150.06C59 393 59 395 58.42 399.5c-2.73 14.11-7.51 39-10 51.91a24 24 0 008 22.92l.46.39A24.34 24.34 0 0072 480a23.42 23.42 0 009-1.79l53.51-20.65a8.05 8.05 0 015.72 0c21.07 7.84 43 12 63.78 12a176 176 0 0074.91-16.66c5.46-2.56 14-5.34 19-11.12a15 15 0 001.95-16.39z"/></svg>
 <p class="text-sm">Chat</p>
 </div>
 <div class="bg-slate-200 rounded-lg w-full h-[86%] px-4 py-6 overflow-y-scroll">

 <div class="flex flex-col gap-y-12">
 <div v-for="(item, index) in currentChatSession" class="flex flex-col gap-y-6">
 <div class="flex flex-col gap-y-2 text-sm">
 <img class="h-6 w-6 my-auto rounded-full" src="https://avatars.githubusercontent.com/u/91587352?v=4">
 <p class="my-auto">{{item.message}}</p>
 </div>

 <div class="flex flex-col gap-y-2 text-sm" >
 <img class="h-6 w-6 my-auto rounded-full bg-black" >
 <div class="dot-container mt-2 -ml-2 scale-[0.6]" v-if="loadingStart && index==currentChatSession.length-1">
 <div class="dot"></div>
 <div class="dot"></div>
 <div class="dot"></div>
</div>

 <p class="my-auto mt-1 typewriter" v-if="item.response!=''">{{item.response}}</p>
 <a v-if="item.response!=''" :href="allFiles.find(file => file.metaData.name === item.file)?.URL" class="mt-1 group rounded-lg text-sm py-2 flex flex-row gap-x-1 w-fit px-2 rounded-md bg-slate-300 transition duration-200 hover:scale-[1.01]">
 <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4 my-auto fill-slate-400" viewBox="0 0 512 512"><path d="M428 224H288a48 48 0 01-48-48V36a4 4 0 00-4-4h-92a64 64 0 00-64 64v320a64 64 0 0064 64h224a64 64 0 0064-64V228a4 4 0 00-4-4z"/><path d="M419.22 188.59L275.41 44.78a2 2 0 00-3.41 1.41V176a16 16 0 0016 16h129.81a2 2 0 001.41-3.41z"/></svg> 
 <p class="text-xs my-auto">{{item.file}}</p>

 </a>

 

 
 </div> 


 </div>
 </div>



 </div>


 <div class="bg-slate-200 rounded-lg w-full h-[14%] flex flex-col px-2 py-2">
 
 <div class="flex flex-row gap-x-1">


 <div class="flex flex-row gap-x-1" v-for="(item, index) in searchArrFilters">
 <div class="group rounded-lg text-sm py-[5px] flex flex-row gap-x-1 w-fit px-2 rounded-md bg-slate-300">
 <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4 my-auto fill-slate-400" viewBox="0 0 512 512"><path d="M428 224H288a48 48 0 01-48-48V36a4 4 0 00-4-4h-92a64 64 0 00-64 64v320a64 64 0 0064 64h224a64 64 0 0064-64V228a4 4 0 00-4-4z"/><path d="M419.22 188.59L275.41 44.78a2 2 0 00-3.41 1.41V176a16 16 0 0016 16h129.81a2 2 0 001.41-3.41z"/></svg> 
 <p class="text-xs my-auto text-slate-500 ">{{ item }}</p> 
 </div>
 </div>

 </div>

 <div>
 <input type="text" id="searchEntry" @keydown.enter="handleEnter" name="searchEntry" v-model="searchEntry" placeholder="Search through files or interact and ask questions..." class="-ml-[0.5rem] mt-1 block w-full h-full bg-slate-200 border-0 text-sm focus:border-slate-200 focus:ring-slate-200">

 </div>

 <div class="flex flex-row justify-end text-slate-400 mt-2">
 <div class="group rounded-lg text-sm py-[5px] flex flex-row gap-x-1 w-fit">
 <p class="text-xs my-auto text-slate-400 ">enter chat</p> 

 <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4 my-auto fill-slate-400" viewBox="0 0 512 512"><path d="M256 64C150.13 64 64 150.13 64 256s86.13 192 192 192 192-86.13 192-192S361.87 64 256 64zm91.31 283.31a16 16 0 01-22.62 0l-42.84-42.83a88.08 88.08 0 1122.63-22.63l42.83 42.84a16 16 0 010 22.62z"/><circle cx="232" cy="232" r="56"/></svg>

 </div>

 </div>

 </div>

 </div>
 </div> 
 


 
 </div>
 </div>

 <div v-if="uploadPopup" class="bg-slate-200 py-6 px-6 h-[20rem] w-[35rem] flex flex-col gap-y-4 rounded-xl absolute absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">

 <div class="flex flex-row justify-between gap-x-2 ">
 <p class="font-bold text-xl my-auto">Upload Memories</p>
 <button class="my-auto transition duration-200 rounded-md hover:bg-slate-300 py-1 px-1" @click="uploadPopup=false; fileToBeUploaded = ref({name:''});"><svg xmlns="http://www.w3.org/2000/svg" class="h-5 my-auto w-5 fill-black" viewBox="0 0 512 512"><path d="M289.94 256l95-95A24 24 0 00351 127l-95 95-95-95a24 24 0 00-34 34l95 95-95 95a24 24 0 1034 34l95-95 95 95a24 24 0 0034-34z"/></svg></button>
 </div>


 <input type="file" ref="uploadFileInput" v-on:change="getImageUploaded();" accept=".py,.java,.js,.cpp,.html,.css,.ts,.rb,.go,.php,.txt,.pdf,.docx,.vue,.csv,.json,.md,.xml,.xlsx" id="uploadFileInput" hidden>
 <label for="uploadFileInput" class="border-[1.5px] cursor-pointer bg-slate-200 border-slate-400 border-dashed h-full w-full gap-y-0 flex flex-col justify-center rounded-lg">
 <div class=" my-auto mx-auto">
 <svg xmlns="http://www.w3.org/2000/svg" class="h-12 w-12 fill-slate-500 mx-auto" viewBox="0 0 512 512"><path d="M473.66 210c-14-10.38-31.2-18-49.36-22.11a16.11 16.11 0 01-12.19-12.22c-7.8-34.75-24.59-64.55-49.27-87.13C334.15 62.25 296.21 47.79 256 47.79c-35.35 0-68 11.08-94.37 32.05a150.07 150.07 0 00-42.06 53 16 16 0 01-11.31 8.87c-26.75 5.4-50.9 16.87-69.34 33.12C13.46 197.33 0 227.24 0 261.39c0 34.52 14.49 66 40.79 88.76 25.12 21.69 58.94 33.64 95.21 33.64h104V230.42l-36.69 36.69a16 16 0 01-23.16-.56c-5.8-6.37-5.24-16.3.85-22.39l63.69-63.68a16 16 0 0122.62 0L331 244.14c6.28 6.29 6.64 16.6.39 22.91a16 16 0 01-22.68.06L272 230.42v153.37h124c31.34 0 59.91-8.8 80.45-24.77 23.26-18.1 35.55-44 35.55-74.83 0-29.94-13.26-55.61-38.34-74.19zM240 448.21a16 16 0 1032 0v-64.42h-32z"/></svg>
 <p class="mx-auto text-xs text-slate-500">Drag & drop or click to choose files</p>
 </div>
 </label>



 <div class="w-full border-[1.5px] border-slate-400 h-[3rem] rounded-md">
 <div class="group rounded-lg text-sm h-[2rem] px-3 flex flex-row gap-x-2 w-full">
 <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-8 my-auto fill-slate-400" viewBox="0 0 512 512"><path d="M428 224H288a48 48 0 01-48-48V36a4 4 0 00-4-4h-92a64 64 0 00-64 64v320a64 64 0 0064 64h224a64 64 0 0064-64V228a4 4 0 00-4-4z"/><path d="M419.22 188.59L275.41 44.78a2 2 0 00-3.41 1.41V176a16 16 0 0016 16h129.81a2 2 0 001.41-3.41z"/></svg> 
 <p class="text-sm my-auto text-slate-500 line-clamp-[1]">{{ fileToBeUploaded.name }}</p> 
 </div>
 </div>

<div class="flex w-full justify-end">
 <button @click="uploadFileToDB()" class="flex flex-row bg-gradient-to-b to-[#17183B] from-[#A11692] rounded-lg transition duration-200 hover:opacity-85 h-[2.75rem] transition duration-200 hover:scale-[1.01] w-full ">

 <div class="gap-x-2 mx-auto my-auto w-full h-full justify-center flex flex-row group">
 <svg xmlns="http://www.w3.org/2000/svg" class="my-auto stroke-white stroke-[60] w-4 h-4 transition duration-200 group-hover:rotate-[20deg]" viewBox="0 0 512 512"><path fill="none" stroke="white" stroke-linecap="round" stroke-linejoin="round" d="M256 112v288M400 256H112"/></svg>
 <p class="text-white my-auto text-sm">Add To Brain</p>
 </div>
</button>
</div>


 </div>

</body>
 </template>
 
 <script setup>
 import {ref} from 'vue';
 import { getStorage, ref as refDB, uploadBytes, listAll, getMetadata, getDownloadURL, deleteObject} from "firebase/storage";
 import {doc, getDoc, getFirestore, updateDoc, deleteDoc, setDoc, arrayUnion, arrayRemove, serverTimestamp, Timestamp, collection, getDocs} from "firebase/firestore"


 const itemSelected = ref("My Storage");
 const uploadPopup = ref(false);
 const fileToBeUploaded = ref({name:""});
 const storage = getStorage();
 const searchArrFilters = ref(["All Files"]);
 const searchEntry = ref("");
 const currentChatSession = ref([]);
 const allFiles = ref([]);
 const loadingStart = ref(true);
 const db = getFirestore();


const listRef = refDB(storage, ''); 
listAll(listRef)
 .then((res) => {
 res.items.forEach((itemRef) => {
 getMetadata(itemRef).then((metadata)=> {
 getDownloadURL(itemRef).then((urlToPush)=> {
 allFiles.value.push({'fileRef':itemRef, 'metaData':metadata, 'URL':urlToPush})
 })
 })
 });

 res.prefixes.forEach((folderRef) => {
 console.log("Subfolder:", folderRef.fullPath);
 });
 })
 .catch((error) => {
 console.error("Error listing files:", error);
 });


 const vNav = ref([
 {name:"My Storage"},

 
 ])



const deleteFile = async (fileUrl) => {
 const fileRef = refDB(storage, fileUrl);
 deleteObject(fileRef)
 .then(() => {
 console.log("File deleted successfully!");
 allFiles.value.splice(allFiles.value.findIndex(file => file.metaData.name === fileUrl), 1);
 const response = fetch('http://127.0.0.1:5000/update_vector_db');
 })
 .catch((error) => {
 console.error("Error deleting file:", error);
 });

}


 const getImageUploaded = () => { 
 fileToBeUploaded.value['name'] = uploadFileInput.files[0].name;
 fileToBeUploaded.value['file'] = uploadFileInput.files[0];
 }
 
 const uploadFileToDB = () => {
 const fileRef = refDB(storage, fileToBeUploaded.value.name);
 uploadBytes(fileRef, fileToBeUploaded.value['file']).then((snapshot) => {
 fileToBeUploaded.value = {'name':''};
 uploadPopup.value = false;
 getDownloadURL(snapshot.ref).then((urlToPush)=> {
 allFiles.value.push({'fileRef':snapshot.ref, 'metaData':snapshot.metadata, 'URL':urlToPush})
 const response = fetch('http://127.0.0.1:5000/update_vector_db');
 })
 
});
 }


 let index = 0;

function typeWriter(text, loadingMessageIndex, updatedFileName) {
 const speed = 50; // milliseconds per word (adjust as necessary)
 
 // Split the text into words
 const words = text.split(' ');
 
 // If there are more words to type, continue the animation
 if (index < words.length) {
 currentChatSession.value[loadingMessageIndex].response += words[index] + ' ';
 index++;
 
 // Use the function reference inside setTimeout, not the result of calling the function
 setTimeout(() => typeWriter(text, loadingMessageIndex, updatedFileName), speed);
 } else {
 // Once typing is done, update the file name
 currentChatSession.value[loadingMessageIndex].file = updatedFileName;
 }
}






 const handleEnter = async () => {
 // Step 1: Push the user's message with an empty response and file
 currentChatSession.value.push({
 'message': searchEntry.value,
 'response': '',
 'file': '',
 });


 const msgToSend = searchEntry.value;
 const loadingMessageIndex = currentChatSession.value.length - 1; // Get the index of the last message
 loadingStart.value = true;
 searchEntry.value = "";
 const response = await fetch('http://127.0.0.1:5000/search?query=' + encodeURIComponent(msgToSend));
 const data = await response.json();
 const updatedFileName = data["file_name"].split("/")[data["file_name"].split("/").length - 1];
 currentChatSession.value[loadingMessageIndex] = {
 'message': msgToSend,
 'response': '',
 'file': updatedFileName, // Update the file name
 };

typeWriter(data['answer'], loadingMessageIndex, updatedFileName);
 loadingStart.value = false;
};


 </script>
 
 <style scoped>
/* The container for the 3 dots */
.dot-container {
 display: inline-block;
 position: relative;
 width: 50px;
 height: 16px;
}

/* Individual dot style */
.dot {
 width: 8px;
 height: 8px;
 border-radius: 50%;
 background-color: #000;
 position: absolute;
 animation: dotAnimation 1.4s infinite ease-in-out;
}

/* Positioning the dots in the container */
.dot:nth-child(1) {
 left: 0;
 animation-delay: 0s;
}

.dot:nth-child(2) {
 left: 18px;
 animation-delay: 0.2s;
}

.dot:nth-child(3) {
 left: 36px;
 animation-delay: 0.4s;
}

/* The animation for the dots */
@keyframes dotAnimation {
 0% {
 opacity: 0;
 transform: scale(0);
 }
 50% {
 opacity: 1;
 transform: scale(1);
 }
 100% {
 opacity: 0;
 transform: scale(0);
 }
}





 </style>
